import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method.base import BugLocalizationMethod
from dotenv import load_dotenv
import os
from method.prompt import PromptGenerator
from method.models import OpenAILocalizerResponse
# LLMClientGenerator not needed for local-only inference
from dataset.utils import get_token_count, chunk_code_files, estimate_prompt_tokens, get_logger
import json
import re
from typing import Optional, Union

# Import Unsloth libraries
# Unsloth provides optimized fine-tuning and inference for LLMs with significant memory savings
try:
    import unsloth
    from unsloth import FastLanguageModel, FastModel
    from transformers import TextStreamer
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    # Create dummy classes to avoid NameError
    FastLanguageModel = None
    FastModel = None
    TextStreamer = None
    torch = None
    UNSLOTH_AVAILABLE = False

logger = get_logger(__name__)

class OpenSourceLocalizer(BugLocalizationMethod):
    def __init__(self, 
                 model="gpt-oss", 
                 device=None,
                 max_seq_length=16384, 
                 max_new_tokens=4096,
                 dtype=None,
                 load_in_4bit=True):
        """
        Initialize Unsloth-based localizer for optimized local inference.
        
        Args:
            model (str): Model name/type (e.g., "gpt-oss", "openai-free")
            device (str): Device for local inference ("cuda", "cpu", or None for auto)
            max_seq_length (int): Maximum sequence length for model context
            max_new_tokens (int): Maximum new tokens to generate
            dtype: Data type (None for auto detection)
            load_in_4bit (bool): Whether to use 4-bit quantization for memory efficiency
        """
        super().__init__()
        load_dotenv()
        
        self.model = model
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        
        # Model mapping for Unsloth (4bit pre-quantized models for faster downloading and no OOMs)
        self.unsloth_model_mapping = {
            "gpt-oss": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "gpt-oss-120b": "unsloth/gpt-oss-120b-unsloth-bnb-4bit", 
            "openai-free": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "gpt-oss-20b": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "gpt-oss-20b-mxfp4": "unsloth/gpt-oss-20b",
            "gpt-oss-120b-mxfp4": "unsloth/gpt-oss-120b",  
            "qwen3_06b": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
            "qwen3_1.4b": "unsloth/Qwen3-1.4B-unsloth-bnb-4bit"
        }
        
        # Initialize components
        self.prompt_generator = PromptGenerator()
        self.unsloth_model = None
        self.unsloth_tokenizer = None
        # Separate model for structured output extraction using Qwen
        self.extractor_model = None
        self.extractor_tokenizer = None
        
        # Set up device
        if device is None:
            self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
            
        logger.info(f"Unsloth localizer initialized: device={self.device}, model={self.model}")
        
        # Check if Unsloth is available
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth not available. Install with: "
                "pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
            )
        
        # Initialize local model
        try:
            self._initialize_unsloth_model()
            logger.info("Unsloth model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Unsloth model: {e}")
            logger.error("Unsloth localizer requires a working local model setup")
            raise e
            
        # Initialize Qwen extractor model for structured output
        try:
            self._initialize_qwen_extractor()
            logger.info("Qwen extractor model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen extractor model: {e}")
            logger.warning("Structured output may be less reliable without Qwen extractor")
            # Don't raise error here, allow main model to work even if extractor fails

    def _initialize_unsloth_model(self):
        """Initialize the Unsloth model for optimized inference."""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available. Install with: pip install unsloth")
        
        model_name = self._get_unsloth_model_name(self.model)
        logger.info(f"Loading Unsloth model: {model_name}")
        
        try:
            # Load model and tokenizer using Unsloth's FastLanguageModel
            self.unsloth_model, self.unsloth_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                dtype=self.dtype,  # None for auto detection
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,  # Use 4-bit quantization for memory efficiency
                full_finetuning=False,  # We're doing inference, not full finetuning
            )
            
            # Get PEFT model for optimized inference
            self.unsloth_model = FastLanguageModel.get_peft_model(
                self.unsloth_model,
                r=8,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,  # 0 is optimized
                bias="none",  # "none" is optimized
                use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized gradient checkpointing
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            logger.info(f"Successfully loaded Unsloth model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Unsloth model {model_name}: {e}")
            raise e

    def _initialize_qwen_extractor(self):
        """Initialize the Qwen model for structured output extraction."""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available for Qwen extractor")
        
        qwen_model_name = "unsloth/Qwen3-1.4B-unsloth-bnb-4bit"
        logger.info(f"Loading Qwen extractor model: {qwen_model_name}")
        
        try:
            # Load Qwen model using FastModel for structured output extraction
            self.extractor_model, self.extractor_tokenizer = FastModel.from_pretrained(
                model_name=qwen_model_name,
                max_seq_length=2055,  
                load_in_4bit=True,   
                load_in_8bit=False,  
                full_finetuning=False,
            )
            
            logger.info(f"Successfully loaded Qwen extractor model: {qwen_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen extractor model {qwen_model_name}: {e}")
            raise e

    def _get_unsloth_model_name(self, model_type: str) -> str:
        """Map simplified model names to Unsloth model names."""
        return self.unsloth_model_mapping.get(model_type, model_type)

    def _generate_with_unsloth_model(self, prompt: str) -> str:
        """Generate response using Unsloth model."""
        if not self.unsloth_model or not self.unsloth_tokenizer:
            raise RuntimeError("Unsloth model not initialized")
        
        try:
            # Create messages format for chat template
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            # Apply chat template and prepare inputs
            inputs = self.unsloth_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort="medium",  # Set reasoning effort to medium for balance
            ).to(self.unsloth_model.device)
            
            # Generate response
            output = self.unsloth_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.unsloth_tokenizer.eos_token_id,
                eos_token_id=self.unsloth_tokenizer.eos_token_id,
            )
            
            # Decode the generated text (remove input prompt)
            generated_text = self.unsloth_tokenizer.decode(
                output[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
                
        except Exception as e:
            logger.error(f"Unsloth generation failed: {e}")
            raise e

    def _extract_structured_output_with_qwen(self, decoded_text: str, json_schema: str) -> str:
        """Extract structured JSON output using Qwen model."""
        if not self.extractor_model or not self.extractor_tokenizer:
            logger.warning("Qwen extractor not available, falling back to regex extraction")
            return decoded_text
        
        try:
            # Create message for Qwen extractor
            messages = [
                {"role": "user", 
                 "content": f"""
                 You will receive messages from another language model that already produced the answer 
                 Your task is to construct JSON without any explanation or extra output using the answer by other LLM and JSON schema

                 JSON Schema:
                {json_schema}

                 Answer by other LLM: 
                 {decoded_text}
                 
                 """}
            ]

            # Apply chat template
            text = self.extractor_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # Must add for generation
            )

            # Generate structured output
            inputs = self.extractor_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.extractor_model.device)
            
            final_output = self.extractor_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.extractor_tokenizer.eos_token_id,
                eos_token_id=self.extractor_tokenizer.eos_token_id,
            )

            # Decode the response
            answer = self.extractor_tokenizer.decode(
                final_output[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Extract content after </think> tags if present
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            
            # Remove the input prompt from the answer
            if text in answer:
                answer = answer.replace(text, "").strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Qwen extraction failed: {e}")
            logger.warning("Falling back to original text")
            return decoded_text

    def _generate_structured_with_unsloth_model(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        """Generate structured response using two-step process: Unsloth + Qwen extraction."""
        try:
            # Step 1: Generate response with main Unsloth model (no strict formatting requirements)
            response_text = self._generate_with_unsloth_model(prompt)
            logger.info("Generated initial response with Unsloth model")
            
            # Step 2: Extract structured JSON using Qwen extractor
            schema = text_format.model_json_schema()
            schema_description = json.dumps(schema, indent=2)
            
            # Use Qwen to extract structured output
            extracted_json = self._extract_structured_output_with_qwen(response_text, schema_description)
            
            # Clean and parse JSON
            extracted_json = extracted_json.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', extracted_json, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = extracted_json
            
            # Parse and validate
            try:
                response_data = json.loads(json_text)
                result = text_format(**response_data)
                logger.info("Successfully extracted structured output using Qwen")
                return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON from Qwen extractor: {e}")
                logger.warning(f"Extracted JSON: {extracted_json}")
                
                # Fallback: try to parse original response directly
                logger.info("Attempting fallback JSON extraction from original response")
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    fallback_json = json_match.group(0)
                    try:
                        response_data = json.loads(fallback_json)
                        return text_format(**response_data)
                    except (json.JSONDecodeError, ValueError):
                        pass
                
                # Final fallback: create a basic response structure
                logger.warning("Creating fallback response structure")
                return text_format(
                    candidate_files=[]
                )
                
        except Exception as e:
            logger.error(f"Structured generation failed with Unsloth model: {e}")
            raise e

    def invoke(self, prompt: str, model_type: Optional[str] = None) -> str:
        """Generate text response using Unsloth model only."""
        model_type = model_type or self.model
        
        try:
            return self._generate_with_unsloth_model(prompt)
        except Exception as e:
            logger.error(f"Unsloth generation failed: {e}")
            logger.error("Unsloth localizer only supports local inference")
            logger.error("Consider using OpenAIFreeLocalizer for remote inference fallback")
            raise RuntimeError(f"Unsloth model generation failed: {e}")

    def invoke_structured(self, prompt: str, text_format, model: Optional[str] = None) -> OpenAILocalizerResponse:
        """Generate structured response using Unsloth model only."""
        model = model or self.model
        
        try:
            return self._generate_structured_with_unsloth_model(prompt, text_format)
        except Exception as e:
            logger.error(f"Unsloth structured generation failed: {e}")
            logger.error("Unsloth localizer only supports local inference")
            logger.error("Consider using OpenAIFreeLocalizer for remote inference fallback")
            raise RuntimeError(f"Unsloth structured generation failed: {e}")

    def localize(self, bug):
        """
        Localize bugs using Unsloth optimized local inference.
        
        Args:
            bug: BugInstance object
        """
        
        # Calculate available tokens based on model's context size
        available_input_tokens = self.max_seq_length - self.max_new_tokens - 250
        max_prompt_tokens = available_input_tokens
        
        # Count tokens in the full bug instance
        full_prompt = self.prompt_generator.generate_openai_prompt(bug)
        total_tokens = get_token_count(full_prompt, model="gpt-4o")
        
        # If prompt fits within limits, process directly
        if total_tokens <= max_prompt_tokens:
            logger.info(f"Processing bug directly ({total_tokens} tokens)")
            response = self.invoke_structured(
                prompt=full_prompt,
                text_format=OpenAILocalizerResponse,
                model=self.model
            )
            return response
        
        # If bug report is very long, summarize it first
        # bug_report_tokens = get_token_count(bug.bug_report, model="gpt-4o")
        # if bug_report_tokens > max_prompt_tokens // 2:
        #     logger.info(f"Summarizing long bug report ({bug_report_tokens} tokens)")
        #     summary_prompt = self.prompt_generator.generate_openai_report_summarizer_prompt(bug)
        #     bug.bug_report = self.invoke(summary_prompt, model_type=self.model)
        
        # Check if code files need chunking
        code_files_tokens = get_token_count("\n\n".join(bug.code_files), "gpt-4o")

        max_chunk_tokens = max_prompt_tokens // 4

        if code_files_tokens <= max_chunk_tokens:
            # Try again with potentially summarized bug report
            prompt = self.prompt_generator.generate_openai_prompt(bug)
            response = self.invoke_structured(
                prompt=prompt,
                text_format=OpenAILocalizerResponse,
                model=self.model
            )
            return response
        
        # Need to chunk code files
        logger.info(f"Chunking code files ({code_files_tokens} tokens > {max_chunk_tokens} limit)")
        chunks = chunk_code_files(bug.code_files, max_chunk_tokens, "gpt-4o")
    
        chunk_responses = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} files...")
            
            # Generate prompt for this chunk
            chunk_prompt = self.prompt_generator.generate_openai_prompt(bug, chunk)
            
            # Final safety check
            chunk_tokens = get_token_count(chunk_prompt, model="gpt-4o")
            if chunk_tokens > max_prompt_tokens:
                logger.warning(f"Chunk {i+1} estimated at {chunk_tokens} tokens, may exceed model limits")
                continue
            
            # Get response for this chunk
            try:
                chunk_response = self.invoke_structured(
                    prompt=chunk_prompt,
                    text_format=OpenAILocalizerResponse,
                    model=self.model
                )
                chunk_responses.append(chunk_response)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Continue with other chunks
                continue
        
        # If we only have one successful response, return it
        if len(chunk_responses) == 1:
            return chunk_responses[0]
        
        # If we have multiple responses, aggregate them
        elif len(chunk_responses) > 1:
            return self._aggregate_chunk_responses(bug, chunk_responses)
        
        # If no chunks were successfully processed, return empty response
        else:
            logger.warning("All chunks failed, returning empty response")
            return OpenAILocalizerResponse(candidate_files=[])

    def _aggregate_chunk_responses(self, bug, chunk_responses):
        """
        Aggregate responses from multiple chunks into a final response.
        
        Args:
            bug: BugInstance object
            chunk_responses: List of OpenAILocalizerResponse objects
        """
        logger.info(f"Aggregating {len(chunk_responses)} chunk responses...")
        
        # Convert structured responses to text for aggregation
        response_texts = []
        for i, response in enumerate(chunk_responses):
            # Convert structured response to text representation
            response_text = f"Files analyzed: {getattr(response, 'files', 'N/A')}\n"
            response_text += f"Reasoning: {getattr(response, 'reasoning', 'N/A')}"
            response_texts.append(response_text)
        
        # Generate aggregation prompt
        aggregation_prompt = self.prompt_generator.generate_chunk_aggregation_prompt(bug, response_texts)
        
        # Get final aggregated response
        try:
            final_response = self.invoke_structured(
                prompt=aggregation_prompt,
                text_format=OpenAILocalizerResponse,
                model=self.model
            )
            return final_response
            
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            # Fall back to the first successful chunk response
            return chunk_responses[0]

    def cleanup(self):
        """Clean up Unsloth model resources."""
        if self.unsloth_model is not None:
            # Clean up main model GPU memory
            del self.unsloth_model
            del self.unsloth_tokenizer
            logger.info("Cleaned up Unsloth main model resources")
            
        if self.extractor_model is not None:
            # Clean up extractor model GPU memory
            del self.extractor_model
            del self.extractor_tokenizer
            logger.info("Cleaned up Qwen extractor model resources")
            
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
