import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method.base import BugLocalizationMethod
from openai import OpenAI
from dotenv import load_dotenv
import os
from method.prompt import PromptGenerator
from method.models import OpenAILocalizerResponse

class OpenAILocalizer(BugLocalizationMethod):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def localize(self, bug):
        prompt = PromptGenerator.generate_openai_prompt(bug)
        response = self.client.responses.parse(
            model="gpt-4o",
            input=[
                # {
                #     "role" : "system",
                #     "content" : OPENAI_LOCALIZER_SYSTEM_PROMPT
                # },
                {
                    "role": "user", 
                    "content": prompt
                }
                ],
            text_format=OpenAILocalizerResponse
        )
        return response.output_parsed
