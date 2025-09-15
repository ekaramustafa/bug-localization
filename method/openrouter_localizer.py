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
    def __init__(self, model="qwen/qwen-2.5-coder-32b-instruct", max_tokens=4096, temperature=0.7):
        super().__init__()
        load_dotenv()
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Model mapping for user-friendly names to OpenRouter model IDs
        self.model_mapping = {
            "qwen-coder-32b": "qwen/qwen-2.5-coder-32b-instruct",
            "deepseek-coder": "deepseek/deepseek-coder",
            "codellama-34b": "codellama/codellama-34b-instruct",
            "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
            "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
            "gpt-oss-20b": "openai/gpt-oss-20b:free"
        }
        
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
        
        logger.info(f"OpenRouterLocalizer initialized with model: {self.model}")
    
    def _make_api_request(self, prompt: str, structured: bool = False, response_format=None) -> str:
        """Make API request to OpenRouter using OpenAI client with proper error handling"""
        try:
            # Map user-friendly model name to OpenRouter model ID
            model_id = self.model_mapping.get(self.model, self.model)
            
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
            
            # Make the API call
            completion = self.client.chat.completions.create(**request_params)
            
            # Log API response metadata
            if hasattr(completion, 'usage') and completion.usage:
                logger.info(f"API usage - prompt_tokens: {completion.usage.prompt_tokens}, "
                           f"completion_tokens: {completion.usage.completion_tokens}, "
                           f"total_tokens: {completion.usage.total_tokens}")
            
            # Extract response content
            response_content = completion.choices[0].message.content
            
            if not response_content:
                logger.warning("API returned empty response content")
                return ""
            
            logger.info(f"API request successful, response length: {len(response_content)} characters")
            logger.debug(f"Response content preview: {response_content[:200]}...")
            return response_content
            
        except Exception as e:
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
    
    def cleanup(self):
        """Cleanup resources (minimal for API-based approach)"""
        logger.info("OpenRouterLocalizer cleanup completed")
    
    def __del__(self):
        """Automatic cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass
    
    def localize(self, bug):
        """Main bug localization method - placeholder for now"""
        # This will be implemented in later tasks
        logger.info("Localize method called - implementation pending")
        return OpenAILocalizerResponse(candidate_files=[])