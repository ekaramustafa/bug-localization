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
    from unsloth import FastLanguageModel
    from transformers import TextStreamer
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    # Create dummy classes to avoid NameError
    FastLanguageModel = None
    TextStreamer = None
    torch = None
    UNSLOTH_AVAILABLE = False

logger = get_logger(__name__)

class OpenSourceLocalizer(BugLocalizationMethod):
    def __init__(self, 
                 model="gpt-oss", 
                 device=None,
                 max_seq_length=1024,
                 max_new_tokens=256,
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
            "qwen3_6b": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
        }
        
        # Initialize components
        self.prompt_generator = PromptGenerator()
        self.unsloth_model = None
        self.unsloth_tokenizer = None
        
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

    def _generate_structured_with_unsloth_model(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        """Generate structured response using Unsloth model."""
        # Create structured prompt
        schema = text_format.model_json_schema()
        schema_description = json.dumps(schema, indent=2)
        
        structured_prompt = f"""{prompt}

        Please respond with a valid JSON object that matches this exact schema:
        {schema_description}

        Respond ONLY with the JSON object, no additional text."""

        try:
            # Generate response
            response_text = self._generate_with_unsloth_model(structured_prompt)
            
            # Clean and parse JSON
            response_text = response_text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                json_text = response_text
            
            # Parse and validate
            try:
                response_data = json.loads(json_text)
                return text_format(**response_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON from Unsloth model: {e}")
                logger.warning(f"Raw response: {response_text}")
                raise e
                
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

    def localize(self, bug, max_prompt_tokens=120000, max_chunk_tokens=60000):
        """
        Localize bugs using Unsloth optimized local inference.
        
        Args:
            bug: BugInstance object
            max_prompt_tokens: Maximum total tokens per prompt
            max_chunk_tokens: Maximum tokens per code file chunk
        """
        
        # Count tokens in the bug report using the utility function
        token_count = get_token_count(bug.bug_report, model="gpt-4o")  # Use gpt-4o for tokenization

        # With large context, we can handle much larger bug reports before summarizing
        if token_count > max_prompt_tokens:
            prompt = self.prompt_generator.generate_openai_report_summarizer_prompt(bug)
            bug_report = self.invoke(prompt, model_type=self.model)
            # Update the bug object with the summarized report
            bug.bug_report = bug_report
        
        # Check if code files need chunking
        code_files_tokens = get_token_count("\n\n".join(bug.code_files), "gpt-4o")
        
        if code_files_tokens <= max_chunk_tokens:
            # No chunking needed - process normally
            prompt = self.prompt_generator.generate_openai_prompt(bug)
            
            response = self.invoke_structured(
                prompt=prompt,
                text_format=OpenAILocalizerResponse,
                model=self.model
            )
            return response
        
        else:
            # Need chunking - split code files and process each chunk
            logger.info(f"Code files require chunking: {code_files_tokens} tokens > {max_chunk_tokens} limit")
            
            # Split code files into manageable chunks
            chunks = chunk_code_files(bug.code_files, max_chunk_tokens, "gpt-4o")
            
            chunk_responses = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} files...")
                
                # Estimate total prompt tokens for this chunk
                estimated_tokens = estimate_prompt_tokens(bug, chunk, "gpt-4o")
                
                if estimated_tokens > max_prompt_tokens:
                    logger.warning(f"Chunk {i+1} estimated at {estimated_tokens} tokens, may exceed model limits")
                
                # Generate prompt for this chunk
                chunk_prompt = self.prompt_generator.generate_openai_prompt(bug, chunk)
                
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
            
            # If no chunks were successfully processed, fall back to original approach
            else:
                logger.warning("All chunks failed, falling back to original approach")
                prompt = self.prompt_generator.generate_openai_prompt(bug)
                return self.invoke_structured(
                    prompt=prompt,
                    text_format=OpenAILocalizerResponse,
                    model=self.model
                )
    
    def _aggregate_chunk_responses(self, bug, chunk_responses):
        """Aggregate multiple chunk responses into a single response."""
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
            # Clean up GPU memory
            del self.unsloth_model
            del self.unsloth_tokenizer
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleaned up Unsloth model resources")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
