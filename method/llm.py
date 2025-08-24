import requests
import os 
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from typing import Type, Any
from pydantic import BaseModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.utils import get_logger
load_dotenv()

logger = get_logger(__name__)

class LLMClientGenerator:
    def __init__(self, api_key: str = None, use_openrouter: bool = False, openrouter_api_key: str = None, use_huggingface: bool = False):
        self.use_openrouter = use_openrouter
        self.use_huggingface = use_huggingface
        
        # Model name mapping for OpenRouter
        self.model_mapping = {
            "qwen3": "qwen/qwen3-coder:free",
            "gpt-4o": "openai/gpt-4o",
            "gpt-4": "openai/gpt-4-turbo",
            "gpt-oss": "openai/gpt-oss-20b:free",
            "openai-free": "openai/gpt-oss-20b:free",
            "gpt-oss-hf": "openai/gpt-oss-20b",  # For Hugging Face local inference
            "gpt-oss-120b-hf": "openai/gpt-oss-120b",  # For Hugging Face local inference
        }
        
        # Hugging Face model mapping
        self.hf_model_mapping = {
            "gpt-oss": "openai/gpt-oss-20b",
            "gpt-oss-120b": "openai/gpt-oss-120b", 
            "openai-free": "openai/gpt-oss-20b",
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "gpt-oss-hf": "openai/gpt-oss-20b",
            "gpt-oss-120b-hf": "openai/gpt-oss-120b",
        }
        
        if self.use_openrouter:
            if not openrouter_api_key:
                self.api_key = os.getenv("OPENROUTER_API_KEY", None)
            else:
                self.api_key = openrouter_api_key
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
            
            self.llm_web_service_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Bug Localization")
            }
        else:
            if not api_key:
                self.api_key = os.getenv("OPENAI_API_KEY", None)
            else:
                self.api_key = api_key
            
            self.llm_web_service_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            self.client = OpenAI(api_key=self.api_key)

    def _get_model_name(self, model_type: str) -> str:
        """Map simplified model names to full model names for OpenRouter."""
        if self.use_openrouter and model_type in self.model_mapping:
            return self.model_mapping[model_type]
        return model_type
    
    def _pydantic_to_json_schema(self, model_class: Type[BaseModel]) -> dict:
        """Convert a Pydantic model to JSON schema format for OpenRouter."""
        schema = model_class.model_json_schema()
        
        # OpenRouter expects a specific format
        return {
            "name": model_class.__name__.lower(),
            "strict": True,
            "schema": schema
        }

    def invoke(self, prompt, model_type="gpt-4o") -> str:
        # Map model name if using OpenRouter
        full_model_name = self._get_model_name(model_type)
        
        if self.use_openrouter:
            # Use OpenAI client with OpenRouter
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                    "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Bug Localization"),
                },
                model=full_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        else:
            # Use original OpenAI API
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": full_model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post(url, json=payload, headers=self.llm_web_service_headers)
            if response.status_code == 200:
                llm_response = response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
            return llm_response
    

    def invoke_structured(self, prompt, text_format, model="gpt-4o") -> Any:
        # Map model name if using OpenRouter
        full_model_name = self._get_model_name(model)
        
        if self.use_openrouter:
            # Use OpenRouter's structured output format
            json_schema = self._pydantic_to_json_schema(text_format)
            schema_description = json.dumps(json_schema, indent=2)
            structured_prompt = f"""{prompt}

            Please respond with a valid JSON object that matches this exact schema:
            {schema_description}

            Respond ONLY with the JSON object, no additional text."""
            
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                        "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Bug Localization"),
                    },
                    model=full_model_name,
                    messages=[
                        {"role": "user", "content": structured_prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": json_schema
                    }
                )
                
                # Parse the JSON response and convert to Pydantic model
                response_content = completion.choices[0].message.content
                
                logger.debug(f"Raw response content: {response_content}")
                
                # Clean up the response content if needed
                response_content = response_content.strip()
                
                # Parse JSON and convert to Pydantic model
                try:
                    response_data = json.loads(response_content)
                    logger.debug(f"Parsed JSON data: {response_data}")
                    return text_format(**response_data)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error: {json_err}")
                    logger.error(f"Response content that failed to parse: {repr(response_content)}")
                    # Try to extract JSON from response if it's wrapped in markdown or other text
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        try:
                            clean_json = json_match.group(0)
                            response_data = json.loads(clean_json)
                            logger.info(f"Successfully parsed JSON after cleaning: {response_data}")
                            return text_format(**response_data)
                        except json.JSONDecodeError:
                            logger.error("Failed to parse even after cleaning")
                    raise json_err
                except Exception as pydantic_err:
                    logger.error(f"Pydantic validation error: {pydantic_err}")
                    logger.error(f"Data that failed validation: {response_data}")
                    raise pydantic_err
                
            except Exception as e:
                logger.error(f"Structured output failed for {full_model_name}: {e}")
                raise e
        else:
            # Use OpenAI's structured response parsing
            response = self.client.responses.parse(
                model=full_model_name,
                input=[{"role":"user", "content": prompt}],
                text_format=text_format
            )
            return response.output_parsed
    


