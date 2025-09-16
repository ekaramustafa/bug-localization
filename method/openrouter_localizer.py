import os
import sys
import json
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI
from method.base import BugLocalizationMethod
from method.models import OpenAILocalizerResponse
from method.prompt import PromptGenerator
from dataset.utils import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)


class DirectorySelectionResponse(BaseModel):
    selected_directory: Optional[str] = None
    selected_files: List[str] = []


class OpenRouterLocalizer(BugLocalizationMethod):
    def __init__(self, model="qwen-coder-32b", max_tokens=4096, temperature=0.7):
        super().__init__()
        load_dotenv()
        
        # Comprehensive model mapping for user-friendly names to OpenRouter model IDs
        # Note: Model availability may vary on OpenRouter, some models may be temporarily unavailable
        self.model_mapping = {
            # Free models (verified working)
            "gpt-oss-20b": "openai/gpt-oss-20b:free",
            "phi-3-mini": "microsoft/phi-3-mini-128k-instruct:free",
            "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct:free",
            
            # Free models (may have limited availability)
            "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
            "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct:free",
            "gemma-7b": "google/gemma-7b-it:free",
            
            # Coding-focused models (recommended for bug localization)
            "qwen-coder-32b": "qwen/qwen-2.5-coder-32b-instruct",
            "qwen-coder-7b": "qwen/qwen-2.5-coder-7b-instruct",
            "deepseek-coder": "deepseek/deepseek-coder",
            "deepseek-coder-v2": "deepseek/deepseek-coder-v2-lite-instruct",
            "codellama-34b": "codellama/codellama-34b-instruct",
            "codellama-13b": "codellama/codellama-13b-instruct",
            "codellama-7b": "codellama/codellama-7b-instruct",
            "starcoder2-15b": "bigcode/starcoder2-15b-instruct",
            "wizardcoder-34b": "wizardlm/wizardcoder-python-34b-v1.0",
            
            # General purpose models
            "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
            "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
            "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
            "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct",
            "nous-hermes-2": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
            "openchat-3.5": "openchat/openchat-7b",
            
            # Premium models (require credits)
            "claude-3-haiku": "anthropic/claude-3-haiku",
            "claude-3-sonnet": "anthropic/claude-3-5-sonnet",
            "gpt-4o": "openai/gpt-4o",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gemini-pro": "google/gemini-pro",
            "gemini-flash": "google/gemini-flash-1.5",
        }
        
        # Default model for bug localization (coding-focused and free)
        self.default_model = "qwen-coder-32b"
        
        # Validate and set model
        self.model = self._validate_and_set_model(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Initialize OpenAI client with OpenRouter configuration
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        self.prompt_generator = PromptGenerator()
        
        # Initialize execution tracking for cleanup statistics
        self._api_call_count = 0
        self._total_tokens_used = 0
        self._successful_requests = 0
        self._failed_requests = 0
        
        logger.info(f"OpenRouterLocalizer initialized with model: {self.model} -> {self.get_model_id()}")
        logger.info(f"Initialization parameters: max_tokens={self.max_tokens}, temperature={self.temperature}")
        logger.debug(f"Available models: {len(self.model_mapping)} total")
    
    def _validate_and_set_model(self, model: str) -> str:
        """Validate model specification and return validated model name
        
        Args:
            model: User-provided model name (can be friendly name or full OpenRouter ID)
            
        Returns:
            Validated model name (friendly name if available, otherwise original)
            
        Raises:
            ValueError: If model is invalid or not supported
        """
        if not model or not isinstance(model, str):
            logger.warning(f"Invalid model specification: {model}, using default: {self.default_model}")
            return self.default_model
        
        model = model.strip()
        
        # Check if it's a friendly name in our mapping
        if model in self.model_mapping:
            logger.info(f"Using friendly model name: {model} -> {self.model_mapping[model]}")
            return model
        
        # Check if it's already a full OpenRouter model ID
        if "/" in model and any(model == full_id for full_id in self.model_mapping.values()):
            # Find the friendly name for this full ID
            for friendly_name, full_id in self.model_mapping.items():
                if full_id == model:
                    logger.info(f"Recognized full model ID, using friendly name: {friendly_name}")
                    return friendly_name
            # If not found in reverse lookup, use as-is but log warning
            logger.warning(f"Using full model ID directly: {model}")
            return model
        
        # Check if it's a partial match (case-insensitive)
        model_lower = model.lower()
        for friendly_name in self.model_mapping.keys():
            if model_lower in friendly_name.lower() or friendly_name.lower() in model_lower:
                logger.info(f"Found partial match: {model} -> {friendly_name}")
                return friendly_name
        
        # If no match found, check if it looks like a valid OpenRouter model ID
        if "/" in model and len(model.split("/")) >= 2:
            logger.warning(f"Unknown OpenRouter model ID, attempting to use: {model}")
            # Store it in mapping for this session
            self.model_mapping[model] = model
            return model
        
        # Invalid model specification
        available_models = list(self.model_mapping.keys())
        error_msg = (f"Invalid model specification: '{model}'. "
                    f"Available models: {', '.join(available_models[:10])}... "
                    f"(total: {len(available_models)} models). "
                    f"Using default: {self.default_model}")
        logger.error(error_msg)
        
        # Use default model instead of raising exception to be more user-friendly
        return self.default_model
    
    def get_model_id(self) -> str:
        """Get the full OpenRouter model ID for the current model
        
        Returns:
            Full OpenRouter model identifier
        """
        return self.model_mapping.get(self.model, self.model)
    
    def get_available_models(self) -> dict:
        """Get dictionary of all available models
        
        Returns:
            Dictionary mapping friendly names to OpenRouter model IDs
        """
        return self.model_mapping.copy()
    
    def list_models_by_category(self) -> dict:
        """Get models organized by category for easier selection
        
        Returns:
            Dictionary with model categories and their models
        """
        categories = {
            "free": [],
            "coding": [],
            "general": [],
            "specialized": []
        }
        
        for friendly_name, model_id in self.model_mapping.items():
            if ":free" in model_id:
                categories["free"].append(friendly_name)
            elif any(keyword in friendly_name for keyword in ["coder", "code", "star"]):
                categories["coding"].append(friendly_name)
            elif any(keyword in friendly_name for keyword in ["llama", "gpt", "claude", "gemini", "mixtral"]):
                categories["general"].append(friendly_name)
            else:
                categories["specialized"].append(friendly_name)
        
        return categories
    
    def _make_api_request(self, prompt: str, structured: bool = False, response_format=None) -> str:
        """Make API request to OpenRouter using OpenAI client with proper error handling"""
        try:
            # Get the validated OpenRouter model ID
            model_id = self.get_model_id()
            
            # Prepare request parameters
            request_params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/bug-localization-tool",
                    "X-Title": "Bug Localization Tool"
                }
            }
            
            # Add structured response format if requested
            if structured and response_format:
                request_params["response_format"] = response_format
            
            logger.info(f"Making API request to model: {model_id}")
            logger.debug(f"Request parameters: max_tokens={self.max_tokens}, temperature={self.temperature}")
            
            # Increment API call counter
            self._api_call_count += 1
            
            # Make the API call
            completion = self.client.chat.completions.create(**request_params)
            
            # Track successful request
            self._successful_requests += 1
            
            # Log API response metadata and track token usage
            if hasattr(completion, 'usage') and completion.usage:
                tokens_used = completion.usage.total_tokens
                self._total_tokens_used += tokens_used
                logger.info(f"API usage - prompt_tokens: {completion.usage.prompt_tokens}, "
                           f"completion_tokens: {completion.usage.completion_tokens}, "
                           f"total_tokens: {tokens_used}")
            
            # Extract response content
            response_content = completion.choices[0].message.content
            
            if not response_content:
                logger.warning("API returned empty response content")
                return ""
            
            logger.info(f"API request successful, response length: {len(response_content)} characters")
            logger.debug(f"Response content preview: {response_content[:200]}...")
            return response_content
            
        except Exception as e:
            # Track failed request
            self._failed_requests += 1
            
            # Handle specific OpenAI client exceptions
            from openai import APIError, RateLimitError, AuthenticationError, APIConnectionError
            
            if isinstance(e, AuthenticationError):
                logger.error("OpenRouter API authentication failed - check your API key")
                raise ValueError("Invalid OpenRouter API key") from e
            elif isinstance(e, RateLimitError):
                logger.error("OpenRouter API rate limit exceeded")
                raise ValueError("Rate limit exceeded - please try again later") from e
            elif isinstance(e, APIConnectionError):
                logger.error("Failed to connect to OpenRouter API")
                raise ValueError("Network connection error - check your internet connection") from e
            elif isinstance(e, APIError):
                error_msg = str(e)
                if "404" in error_msg and "No endpoints found" in error_msg:
                    logger.error(f"Model not available on OpenRouter: {model_id}")
                    raise ValueError(f"Model '{model_id}' is not currently available on OpenRouter. "
                                   f"Try a different model or check OpenRouter's model availability.") from e
                else:
                    logger.error(f"OpenRouter API error: {e}")
                    raise ValueError(f"API error: {e}") from e
            else:
                logger.error(f"Unexpected error during API request: {e}")
                raise ValueError(f"Unexpected error: {e}") from e
    
    def invoke(self, prompt: str, model_type: Optional[str] = None) -> str:
        """Generate text response using OpenRouter API
        
        Args:
            prompt: The input prompt for text generation
            model_type: Optional model type override (not used in this implementation)
            
        Returns:
            Generated text response from the model
        """
        logger.info(f"Starting text generation with model: {self.model}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            # Make API request for regular text completion
            response = self._make_api_request(prompt)
            
            # Log response details
            if response:
                logger.info(f"Text generation successful, response length: {len(response)} characters")
                logger.debug(f"Response preview: {response[:200]}...")
            else:
                logger.warning("Text generation returned empty response")
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def _generate_json_schema(self, pydantic_model):
        """Generate JSON schema from Pydantic model for OpenAI structured output"""
        try:
            # Get the JSON schema from the Pydantic model
            model_schema = pydantic_model.model_json_schema()
            
            # Convert to OpenAI's expected format
            schema_name = pydantic_model.__name__.lower().replace('response', '_response')
            
            # Ensure required fields are properly set
            if 'required' not in model_schema:
                model_schema['required'] = list(model_schema.get('properties', {}).keys())
            
            # Set additionalProperties to False for strict mode
            model_schema['additionalProperties'] = False
            
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": model_schema
                }
            }
        except Exception as e:
            logger.warning(f"Failed to generate JSON schema from Pydantic model: {e}")
            # Fallback to basic schema for OpenAILocalizerResponse
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "bug_localization_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "candidate_files": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["candidate_files"],
                        "additionalProperties": False
                    }
                }
            }

    def invoke_structured(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        """Generate structured JSON response using OpenRouter API with JSON schema
        
        Args:
            prompt: The input prompt for structured generation
            text_format: Pydantic model class for response structure
            
        Returns:
            Structured response parsed into the specified Pydantic model
        """
        logger.info(f"Starting structured response generation with model: {self.model}")
        logger.debug(f"Response format: {text_format.__name__}")
        
        try:
            # Generate JSON schema from Pydantic model
            response_format = self._generate_json_schema(text_format)
            logger.debug(f"Generated JSON schema: {response_format}")
            
            # Make structured API request
            response_content = self._make_api_request(
                prompt, 
                structured=True, 
                response_format=response_format
            )
            
            if not response_content:
                logger.warning("Structured API request returned empty content")
                return self._create_empty_response(text_format)
            
            # Parse JSON response
            try:
                response_data = json.loads(response_content)
                logger.info("Successfully parsed structured JSON response")
                logger.debug(f"Parsed data: {response_data}")
                return text_format(**response_data)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from structured response: {e}")
                logger.debug(f"Raw response content: {response_content[:500]}...")
                raise
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Structured response parsing failed: {e}")
            # Fallback: try to extract JSON from unstructured response
            return self._fallback_structured_parsing(prompt, text_format)
        
        except Exception as e:
            logger.error(f"Structured response generation failed: {e}")
            return self._create_empty_response(text_format)
    
    def _fallback_structured_parsing(self, prompt: str, text_format):
        """Fallback method to extract structured data from unstructured response"""
        try:
            logger.info("Attempting fallback structured parsing with unstructured request")
            response_content = self._make_api_request(prompt)
            
            if not response_content:
                return self._create_empty_response(text_format)
            
            # Look for JSON in the response using regex
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                logger.debug(f"Extracted JSON from unstructured response: {json_text[:200]}...")
                response_data = json.loads(json_text)
                return text_format(**response_data)
            else:
                logger.warning("No JSON found in unstructured response")
                return self._create_empty_response(text_format)
                
        except Exception as e:
            logger.warning(f"Fallback structured parsing failed: {e}")
            return self._create_empty_response(text_format)
    
    def _create_empty_response(self, text_format):
        """Create an empty response of the specified format"""
        try:
            # Try to create empty response based on model fields
            if hasattr(text_format, 'model_fields'):
                empty_data = {}
                for field_name, field_info in text_format.model_fields.items():
                    if field_info.annotation == list or (hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ == list):
                        empty_data[field_name] = []
                    elif field_info.annotation == str:
                        empty_data[field_name] = ""
                    else:
                        empty_data[field_name] = None
                return text_format(**empty_data)
            else:
                # Fallback for older Pydantic versions or unknown structure
                return text_format(candidate_files=[])
        except Exception as e:
            logger.warning(f"Failed to create empty response: {e}")
            # Final fallback - assume it's OpenAILocalizerResponse-like
            return text_format(candidate_files=[])
    
    def test_api_connectivity(self) -> bool:
        """Test basic API connectivity with a simple completion request"""
        try:
            logger.info("Testing OpenRouter API connectivity...")
            
            # Simple test prompt
            test_prompt = "Hello, this is a test. Please respond with 'API connection successful'."
            
            # Make a simple API request
            response = self._make_api_request(test_prompt)
            
            if response and len(response.strip()) > 0:
                logger.info("API connectivity test successful")
                logger.debug(f"Test response: {response[:100]}...")
                return True
            else:
                logger.error("API connectivity test failed - empty response")
                return False
                
        except Exception as e:
            logger.error(f"API connectivity test failed: {e}")
            return False
    
    def check_model_availability(self, model_name: str = None) -> bool:
        """Check if a specific model is available on OpenRouter
        
        Args:
            model_name: Model name to check (uses current model if None)
            
        Returns:
            True if model is available, False otherwise
        """
        if model_name is None:
            model_name = self.model
        
        original_model = self.model
        
        try:
            # Temporarily switch to test model
            self.model = self._validate_and_set_model(model_name)
            model_id = self.get_model_id()
            
            logger.info(f"Checking availability of model: {model_name} -> {model_id}")
            
            # Simple test prompt
            test_prompt = "Test"
            
            # Make a minimal API request
            response = self._make_api_request(test_prompt)
            
            if response:
                logger.info(f"Model {model_name} is available")
                return True
            else:
                logger.warning(f"Model {model_name} returned empty response")
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "not currently available" in error_msg or "404" in error_msg:
                logger.warning(f"Model {model_name} is not available: {e}")
            else:
                logger.error(f"Error checking model {model_name}: {e}")
            return False
        finally:
            # Restore original model
            self.model = original_model
    
    def test_model_compatibility(self, test_models: List[str] = None) -> dict:
        """Test compatibility with different OpenRouter models
        
        Args:
            test_models: List of model names to test (uses default set if None)
            
        Returns:
            Dictionary with test results for each model
        """
        if test_models is None:
            # Test a representative set of models from different categories
            test_models = [
                "gpt-oss-20b",      # Free model
                "qwen-coder-32b",   # Coding model (default)
                "llama-3.1-8b",     # Free general model
                "deepseek-coder",   # Alternative coding model
                "mixtral-8x7b"      # General purpose model
            ]
        
        results = {}
        original_model = self.model
        
        logger.info(f"Testing compatibility with {len(test_models)} models")
        
        for model_name in test_models:
            logger.info(f"Testing model: {model_name}")
            
            try:
                # Temporarily switch to test model
                self.model = self._validate_and_set_model(model_name)
                model_id = self.get_model_id()
                
                # Simple test prompt for code understanding
                test_prompt = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# What does this function do? Respond in one sentence.
"""
                
                # Test basic text generation
                response = self._make_api_request(test_prompt)
                
                if response and len(response.strip()) > 10:
                    results[model_name] = {
                        "status": "success",
                        "model_id": model_id,
                        "response_length": len(response),
                        "response_preview": response[:100] + "..." if len(response) > 100 else response
                    }
                    logger.info(f"Model {model_name} test successful")
                else:
                    results[model_name] = {
                        "status": "failed",
                        "model_id": model_id,
                        "error": "Empty or too short response"
                    }
                    logger.warning(f"Model {model_name} test failed: empty response")
                    
            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "model_id": self.model_mapping.get(model_name, model_name),
                    "error": str(e)
                }
                logger.error(f"Model {model_name} test error: {e}")
        
        # Restore original model
        self.model = original_model
        
        # Log summary
        successful_models = [name for name, result in results.items() if result["status"] == "success"]
        logger.info(f"Model compatibility test completed: {len(successful_models)}/{len(test_models)} models successful")
        
        return results
    
    def cleanup(self):
        """Cleanup resources and log execution statistics
        
        For API-based approach, cleanup is minimal but includes:
        - Clearing client references
        - Logging execution statistics
        - Resetting internal state
        """
        try:
            logger.info("Starting OpenRouterLocalizer cleanup")
            
            # Log execution statistics if available
            if hasattr(self, '_api_call_count'):
                logger.info(f"Total API calls made: {self._api_call_count}")
            if hasattr(self, '_total_tokens_used'):
                logger.info(f"Total tokens used: {self._total_tokens_used}")
            if hasattr(self, '_successful_requests'):
                logger.info(f"Successful requests: {self._successful_requests}")
            if hasattr(self, '_failed_requests'):
                logger.info(f"Failed requests: {self._failed_requests}")
            
            # Clear client reference (helps with garbage collection)
            if hasattr(self, 'client'):
                self.client = None
                logger.debug("OpenAI client reference cleared")
            
            # Clear model mapping and other large objects
            if hasattr(self, 'model_mapping'):
                self.model_mapping.clear()
                logger.debug("Model mapping cleared")
            
            # Clear prompt generator reference
            if hasattr(self, 'prompt_generator'):
                self.prompt_generator = None
                logger.debug("Prompt generator reference cleared")
            
            # Reset API key reference for security
            if hasattr(self, 'api_key'):
                self.api_key = None
                logger.debug("API key reference cleared")
            
            logger.info("OpenRouterLocalizer cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during OpenRouterLocalizer cleanup: {e}")
            # Don't re-raise exceptions in cleanup to avoid masking original errors
    
    def __del__(self):
        """Automatic cleanup on object destruction
        
        Ensures resources are properly cleaned up even if cleanup() is not called explicitly.
        Uses try-except to prevent exceptions during garbage collection.
        """
        try:
            logger.debug("OpenRouterLocalizer destructor called")
            self.cleanup()
        except Exception as e:
            # Silently handle cleanup errors in destructor to avoid issues during garbage collection
            # We can't use logger here as it might already be cleaned up
            pass
    
    def _get_hierarchical_files(self, bug, code_files):
        """Get hierarchical file selection using directory tree exploration
        
        Args:
            bug: BugInstance containing bug report and context
            code_files: List of file paths to organize hierarchically
            
        Returns:
            List of selected file paths based on hierarchical exploration
        """
        if not code_files:
            logger.warning("No code files provided for hierarchical selection")
            return []
        
        logger.info(f"Starting hierarchical file selection with {len(code_files)} files")
        
        # Group files by their directory structure
        file_tree = {}
        for file_path in code_files:
            # Handle both forward and backward slashes
            parts = file_path.replace('\\', '/').split('/')
            current = file_tree
            for part in parts[:-1]:  # All except filename
                if part and part not in current:  # Skip empty parts
                    current[part] = {}
                    current = current[part]
                elif part in current:
                    current = current[part]
            # Add filename to the directory
            if '_files' not in current:
                current['_files'] = []
            current['_files'].append(file_path)
        
        logger.debug(f"Built file tree with {len(file_tree)} root directories")
        
        # Start exploration from root
        selected_files = []
        self._explore_tree(bug, file_tree, selected_files, "")
        
        logger.info(f"Hierarchical selection completed, selected {len(selected_files)} files")
        return selected_files

    def _explore_tree(self, bug, tree, selected_files, current_path, max_depth=5):
        """Recursively explore directory tree to select relevant files
        
        Args:
            bug: BugInstance containing bug report and context
            tree: Dictionary representing directory structure
            selected_files: List to accumulate selected file paths
            current_path: Current directory path being explored
            max_depth: Maximum recursion depth to prevent infinite loops
        """
        if max_depth <= 0:
            logger.debug(f"Maximum exploration depth reached at path: {current_path}")
            return
        
        # Get directories and files at current level
        directories = [k for k in tree.keys() if k != '_files']
        files = tree.get('_files', [])
        
        if not directories and not files:
            logger.debug(f"No directories or files found at path: {current_path}")
            return
        
        logger.debug(f"Exploring path '{current_path}': {len(directories)} directories, {len(files)} files")
        
        # Present options to LLM
        options = []
        if directories:
            options.extend([f"DIR: {d}" for d in directories[:10]])  # Limit to 10 directories
        if files:
            file_names = [os.path.basename(f) for f in files[:20]]  # Limit to 20 files
            options.extend([f"FILE: {name}" for name in file_names])
        
        # Create exploration prompt
        prompt = f"""
You are given a mixed list of directories and files along with bug_report.
Your task is to hierarchically extract the possible directories to get the candidate files.
You must use the provided json_schema to output your result

Bug: {bug.bug_report[:1000]}

Current path: {current_path or 'root'}
Available options:
{chr(10).join(options)}

Select ONE of:
1. Directory name to explore (set selected_directory field)
2. File names to add as candidates (set selected_files field with just names, no "FILE:" prefix)  
3. Leave both fields empty if no relevant options

You must use the provided json schema for your output"""
        
        logger.debug(f"Sending exploration prompt for path: {current_path}")
        
        try:
            # Get structured response from the model
            response = self.invoke_structured(prompt, DirectorySelectionResponse)
            
            # Handle directory selection
            if response.selected_directory:
                logger.debug(f"Model selected directory: {response.selected_directory}")
                for d in directories:
                    if d.lower() == response.selected_directory.lower():
                        next_path = os.path.join(current_path, d) if current_path else d
                        logger.debug(f"Exploring selected directory: {next_path}")
                        self._explore_tree(bug, tree[d], selected_files, next_path, max_depth - 1)
                        return
                logger.warning(f"Selected directory '{response.selected_directory}' not found in available directories")
            
            # Handle file selection
            if response.selected_files:
                logger.debug(f"Model selected files: {response.selected_files}")
                for file_path in files:
                    file_name = os.path.basename(file_path)
                    if any(file_name.lower() == sf.lower() for sf in response.selected_files):
                        selected_files.append(file_path)
                        logger.debug(f"Added file to selection: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error during tree exploration at path '{current_path}': {e}")
            # Continue exploration despite errors

    def localize(self, bug):
        """Main bug localization method with token management and hierarchical selection
        
        Args:
            bug: BugInstance containing bug report, code files, and other context
            
        Returns:
            OpenAILocalizerResponse with candidate files and reasoning
        """
        logger.info(f"Starting bug localization for instance: {bug.instance_id}")
        logger.info(f"Repository: {bug.repo}")
        logger.info(f"Total code files: {len(bug.code_files) if bug.code_files else 0}")
        
        if not bug.code_files:
            logger.warning("No code files provided for localization")
            return OpenAILocalizerResponse(candidate_files=[])
        
        # Import token counting utility
        from dataset.utils import get_token_count
        
        # Calculate available tokens for context (reserve tokens for response)
        # Most OpenRouter models have context windows between 8k-32k tokens
        # We'll use a conservative estimate and reserve space for the response
        max_context_tokens = 8000  # Conservative estimate for most models
        reserved_response_tokens = self.max_tokens  # Reserve space for model response
        available_tokens = max_context_tokens - reserved_response_tokens - 500  # Extra buffer
        
        logger.info(f"Token budget: max_context={max_context_tokens}, "
                   f"reserved_response={reserved_response_tokens}, "
                   f"available={available_tokens}")
        
        # Try direct processing first - generate full prompt and check token count
        logger.info("Attempting direct processing with all files")
        full_prompt = self.prompt_generator.generate_openai_prompt(bug)
        full_prompt_tokens = get_token_count(full_prompt, model="gpt-4o")
        
        logger.info(f"Full prompt token count: {full_prompt_tokens}")
        
        if full_prompt_tokens <= available_tokens:
            logger.info("Full prompt fits within token limit, proceeding with direct processing")
            try:
                response = self.invoke_structured(full_prompt, OpenAILocalizerResponse)
                logger.info(f"Direct processing successful, found {len(response.candidate_files)} candidates")
                return response
            except Exception as e:
                logger.error(f"Direct processing failed: {e}")
                # Fall through to hierarchical approach
        else:
            logger.info(f"Full prompt ({full_prompt_tokens} tokens) exceeds limit ({available_tokens} tokens)")
        
        # Fallback to hierarchical file selection
        logger.info("Using hierarchical file selection to reduce context size")
        try:
            selected_files = self._get_hierarchical_files(bug, bug.code_files)
            
            if not selected_files:
                logger.warning("Hierarchical selection returned no files")
                return OpenAILocalizerResponse(candidate_files=[])
            
            logger.info(f"Hierarchical selection chose {len(selected_files)} files")
            logger.debug(f"Selected files: {selected_files[:10]}...")  # Log first 10 files
            
            # Create prompt with selected files
            selected_prompt = self.prompt_generator.generate_openai_prompt(bug, selected_files)
            selected_prompt_tokens = get_token_count(selected_prompt, model="gpt-4o")
            
            logger.info(f"Selected files prompt token count: {selected_prompt_tokens}")
            
            # If still too large, implement progressive reduction
            if selected_prompt_tokens > available_tokens:
                logger.warning(f"Selected files prompt still too large ({selected_prompt_tokens} tokens)")
                return self._progressive_file_reduction(bug, selected_files, available_tokens)
            
            # Process with selected files
            response = self.invoke_structured(selected_prompt, OpenAILocalizerResponse)
            logger.info(f"Hierarchical processing successful, found {len(response.candidate_files)} candidates")
            return response
            
        except Exception as e:
            logger.error(f"Hierarchical processing failed: {e}")
            # Final fallback - try with minimal context
            return self._minimal_context_fallback(bug, available_tokens)
    
    def _progressive_file_reduction(self, bug, selected_files, available_tokens):
        """Progressively reduce the number of files until prompt fits within token limit
        
        Args:
            bug: BugInstance object
            selected_files: List of selected file paths
            available_tokens: Maximum tokens available for the prompt
            
        Returns:
            OpenAILocalizerResponse with results from reduced file set
        """
        logger.info("Starting progressive file reduction")
        
        from dataset.utils import get_token_count
        
        # Try reducing files in batches
        reduction_steps = [0.8, 0.6, 0.4, 0.2, 0.1]  # Reduce to 80%, 60%, 40%, 20%, 10%
        
        for reduction_factor in reduction_steps:
            reduced_count = max(1, int(len(selected_files) * reduction_factor))
            reduced_files = selected_files[:reduced_count]
            
            logger.info(f"Trying with {reduced_count} files (reduction factor: {reduction_factor})")
            
            reduced_prompt = self.prompt_generator.generate_openai_prompt(bug, reduced_files)
            reduced_tokens = get_token_count(reduced_prompt, model="gpt-4o")
            
            logger.debug(f"Reduced prompt tokens: {reduced_tokens}")
            
            if reduced_tokens <= available_tokens:
                logger.info(f"Found suitable reduction: {reduced_count} files, {reduced_tokens} tokens")
                try:
                    response = self.invoke_structured(reduced_prompt, OpenAILocalizerResponse)
                    logger.info(f"Progressive reduction successful, found {len(response.candidate_files)} candidates")
                    return response
                except Exception as e:
                    logger.error(f"Progressive reduction failed at {reduction_factor}: {e}")
                    continue
        
        logger.warning("Progressive reduction failed at all levels")
        return self._minimal_context_fallback(bug, available_tokens)
    
    def _minimal_context_fallback(self, bug, available_tokens):
        """Final fallback with minimal context when all other approaches fail
        
        Args:
            bug: BugInstance object
            available_tokens: Maximum tokens available for the prompt
            
        Returns:
            OpenAILocalizerResponse with best-effort results
        """
        logger.info("Using minimal context fallback")
        
        from dataset.utils import get_token_count
        
        try:
            # Create a minimal prompt with just the bug report and a few files
            # Take the first few files that might be most relevant based on name patterns
            priority_files = []
            
            # Look for files that might be relevant based on common patterns
            for file_path in bug.code_files[:50]:  # Check first 50 files
                file_name = file_path.lower()
                # Prioritize files with common bug-related patterns
                if any(pattern in file_name for pattern in ['main', 'core', 'base', 'util', 'service', 'controller', 'model']):
                    priority_files.append(file_path)
                if len(priority_files) >= 10:  # Limit to 10 priority files
                    break
            
            # If no priority files found, just take the first few
            if not priority_files:
                priority_files = bug.code_files[:5]
            
            logger.info(f"Minimal context using {len(priority_files)} priority files")
            
            minimal_prompt = self.prompt_generator.generate_openai_prompt(bug, priority_files)
            minimal_tokens = get_token_count(minimal_prompt, model="gpt-4o")
            
            logger.info(f"Minimal prompt tokens: {minimal_tokens}")
            
            if minimal_tokens <= available_tokens:
                response = self.invoke_structured(minimal_prompt, OpenAILocalizerResponse)
                logger.info(f"Minimal context fallback successful, found {len(response.candidate_files)} candidates")
                return response
            else:
                logger.error(f"Even minimal context ({minimal_tokens} tokens) exceeds limit")
                # Return empty response as last resort
                return OpenAILocalizerResponse(candidate_files=[])
                
        except Exception as e:
            logger.error(f"Minimal context fallback failed: {e}")
            return OpenAILocalizerResponse(candidate_files=[])