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

# Import Hugging Face libraries
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        pipeline,
        TextGenerationPipeline
    )
    import torch
    HF_AVAILABLE = True
except ImportError:
    # Create dummy classes to avoid NameError
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None
    TextGenerationPipeline = None
    torch = None
    HF_AVAILABLE = False

logger = get_logger(__name__)

class HuggingFaceLocalizer(BugLocalizationMethod):
    def __init__(self, 
                 model="gpt-oss", 
                 device=None,
                 max_length=4096,
                 temperature=0.7,
                 do_sample=True):
        """
        Initialize HuggingFace localizer for local inference only.
        
        Args:
            model (str): Model name/type (e.g., "gpt-oss", "openai-free")
            device (str): Device for local inference ("cuda", "cpu", or None for auto)
            max_length (int): Maximum generation length for local inference
            temperature (float): Sampling temperature for local inference
            do_sample (bool): Whether to use sampling for local inference
        """
        super().__init__()
        load_dotenv()
        
        self.model = model
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Model mapping for Hugging Face
        self.hf_model_mapping = {
            "gpt-oss": "openai/gpt-oss-20b",
            "gpt-oss-120b": "openai/gpt-oss-120b", 
            "openai-free": "openai/gpt-oss-20b",
            "gpt-oss-20b": "openai/gpt-oss-20b",
            # Fallback options for when GPT-OSS is not available
            "llama3": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "codellama": "codellama/CodeLlama-13b-Instruct-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        }
        
        # Initialize components
        self.prompt_generator = PromptGenerator()
        self.local_pipeline = None
        self.local_tokenizer = None
        
        # Set up device
        if device is None:
            self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
            
        logger.info(f"HuggingFace localizer initialized: device={self.device}, model={self.model}")
        
        # Check if HuggingFace is available
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers not available. Install with: "
                "pip install transformers torch accelerate bitsandbytes"
            )
        
        # Initialize local model
        try:
            self._initialize_local_model()
            logger.info("Local model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            logger.error("HuggingFace localizer requires a working local model setup")
            raise e

    def _initialize_local_model(self):
        """Initialize the local Hugging Face model."""
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers not available. Install with: pip install transformers torch")
        
        model_name = self._get_hf_model_name(self.model)
        logger.info(f"Loading local model: {model_name}")
        
        try:
            # Load tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # For larger models, use additional optimization
            if "120b" in model_name.lower():
                model_kwargs["low_cpu_mem_usage"] = True
                model_kwargs["load_in_8bit"] = True  # Use 8-bit quantization for large models
            
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Create pipeline
            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.local_tokenizer,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            
            logger.info(f"Successfully loaded local model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

    def _get_hf_model_name(self, model_type: str) -> str:
        """Map simplified model names to Hugging Face model names."""
        return self.hf_model_mapping.get(model_type, model_type)

    def _generate_with_local_model(self, prompt: str) -> str:
        """Generate response using local Hugging Face model."""
        if not self.local_pipeline:
            raise RuntimeError("Local model not initialized")
        
        try:
            # Generate response
            outputs = self.local_pipeline(
                prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.local_tokenizer.eos_token_id,
                truncation=True,
                return_full_text=False  # Only return the generated text
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                return generated_text.strip()
            else:
                raise RuntimeError("No output generated from local model")
                
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise e

    def _generate_structured_with_local_model(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        """Generate structured response using local model."""
        # Create structured prompt
        schema = text_format.model_json_schema()
        schema_description = json.dumps(schema, indent=2)
        
        structured_prompt = f"""{prompt}

Please respond with a valid JSON object that matches this exact schema:
{schema_description}

Respond ONLY with the JSON object, no additional text."""

        try:
            # Generate response
            response_text = self._generate_with_local_model(structured_prompt)
            
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
                logger.warning(f"Failed to parse JSON from local model: {e}")
                logger.warning(f"Raw response: {response_text}")
                raise e
                
        except Exception as e:
            logger.error(f"Structured generation failed with local model: {e}")
            raise e

    def invoke(self, prompt: str, model_type: Optional[str] = None) -> str:
        """Generate text response using local model only."""
        model_type = model_type or self.model
        
        try:
            return self._generate_with_local_model(prompt)
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            logger.error("HuggingFace localizer only supports local inference")
            logger.error("Consider using OpenAIFreeLocalizer for remote inference fallback")
            raise RuntimeError(f"Local model generation failed: {e}")

    def invoke_structured(self, prompt: str, text_format, model: Optional[str] = None) -> OpenAILocalizerResponse:
        """Generate structured response using local model only."""
        model = model or self.model
        
        try:
            return self._generate_structured_with_local_model(prompt, text_format)
        except Exception as e:
            logger.error(f"Local structured generation failed: {e}")
            logger.error("HuggingFace localizer only supports local inference")
            logger.error("Consider using OpenAIFreeLocalizer for remote inference fallback")
            raise RuntimeError(f"Local structured generation failed: {e}")

    def localize(self, bug, max_prompt_tokens=120000, max_chunk_tokens=60000):
        """
        Localize bugs using local inference only.
        
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
        """Clean up resources."""
        if self.local_pipeline is not None:
            # Clean up GPU memory
            del self.local_pipeline
            del self.local_tokenizer
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleaned up local model resources")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
