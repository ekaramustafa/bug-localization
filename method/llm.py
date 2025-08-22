import requests
import os 
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

class LLMClientGenerator:
    def __init__(self, api_key: str = None, use_openrouter: bool = False, openrouter_api_key: str = None):
        self.use_openrouter = use_openrouter
        
        # Model name mapping for OpenRouter
        self.model_mapping = {
            "qwen3": "qwen/qwen3-coder:free",
            "gpt-4o": "openai/gpt-4o",
            "gpt-4": "openai/gpt-4-turbo",
            "gpt-oss": "openai/gpt-oss-20b:free",
            "openai-free": "openai/gpt-oss-20b:free",
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
    

    def invoke_structured(self, prompt, text_format, model="gpt-4o") -> str:
        # Map model name if using OpenRouter
        full_model_name = self._get_model_name(model)
        
        response = self.client.responses.parse(
            model=full_model_name,
            input=[{"role":"user", "content": prompt}],
            text_format=text_format
        )
        return response.output_parsed
    


