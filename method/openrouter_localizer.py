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
        """Make API request to OpenRouter using OpenAI client"""
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
            
            # Make the API call
            completion = self.client.chat.completions.create(**request_params)
            
            # Extract response content
            response_content = completion.choices[0].message.content
            
            logger.debug(f"API request successful, response length: {len(response_content)}")
            return response_content
            
        except Exception as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
    
    def invoke(self, prompt: str, model_type: Optional[str] = None) -> str:
        """Generate text response using OpenRouter API"""
        return self._make_api_request(prompt)
    
    def invoke_structured(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        """Generate structured JSON response using OpenRouter API"""
        try:
            # Generate JSON schema for structured output
            response_format = {
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
                            },
                            "reasoning": {"type": "string"}
                        },
                        "required": ["candidate_files"],
                        "additionalProperties": False
                    }
                }
            }
            
            # Make structured API request
            response_content = self._make_api_request(
                prompt, 
                structured=True, 
                response_format=response_format
            )
            
            # Parse JSON response
            response_data = json.loads(response_content)
            return text_format(**response_data)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse structured response: {e}")
            # Fallback: try to extract JSON from unstructured response
            try:
                response_content = self._make_api_request(prompt)
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group(0))
                    return text_format(**response_data)
            except Exception:
                pass
            
            # Final fallback: return empty response
            return text_format(candidate_files=[])
        
        except Exception as e:
            logger.error(f"Structured response generation failed: {e}")
            return text_format(candidate_files=[])
    
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