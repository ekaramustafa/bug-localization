import os
import sys
import json
import re
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from method.base import BugLocalizationMethod
from method.prompt import PromptGenerator
from method.models import OpenAILocalizerResponse
from dataset.utils import get_token_count, chunk_code_files, get_logger

try:
    from unsloth import FastLanguageModel
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    FastLanguageModel = None
    torch = None
    UNSLOTH_AVAILABLE = False

logger = get_logger(__name__)

class OpenSourceLocalizer(BugLocalizationMethod):
    def __init__(self, model="gpt-oss", device=None, max_seq_length=16384, 
                 max_new_tokens=4096, dtype=None, load_in_4bit=True):
        super().__init__()
        load_dotenv()
        
        self.model = model
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        
        self.model_mapping = {
            "gpt-oss": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "gpt-oss-120b": "unsloth/gpt-oss-120b-unsloth-bnb-4bit", 
            "openai-free": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "gpt-oss-20b": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "qwen3_4b_instruct": "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
        }
        
        self.prompt_generator = PromptGenerator()
        self.unsloth_model = None
        self.unsloth_tokenizer = None
        
        self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available. Install with: pip install unsloth")
        
        self._initialize_model()
        logger.info(f"OpenSource localizer initialized: device={self.device}, model={self.model}")

    def _initialize_model(self):
        model_name = self.model_mapping.get(self.model, self.model)
        logger.info(f"Loading model: {model_name}")
        
        self.unsloth_model, self.unsloth_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            dtype=self.dtype,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            full_finetuning=False,
        )
        
        self.unsloth_model = FastLanguageModel.get_peft_model(
            self.unsloth_model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    def _generate_text(self, prompt: str) -> str:
        if not self.unsloth_model or not self.unsloth_tokenizer:
            raise RuntimeError("Model not initialized")
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.unsloth_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="medium",
        ).to(self.unsloth_model.device)
        
        output = self.unsloth_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.unsloth_tokenizer.eos_token_id,
            eos_token_id=self.unsloth_tokenizer.eos_token_id,
        )
        
        generated_text = self.unsloth_tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()

    def invoke(self, prompt: str, model_type: Optional[str] = None) -> str:
        return self._generate_text(prompt)

    def invoke_structured(self, prompt: str, text_format) -> OpenAILocalizerResponse:
        response_text = self._generate_text(prompt)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        json_text = json_match.group(0) if json_match else response_text
        
        try:
            response_data = json.loads(json_text)
            return text_format(**response_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return text_format(candidate_files=[])

    def localize(self, bug):
        available_tokens = self.max_seq_length - self.max_new_tokens - 250
        
        # Try direct processing first
        full_prompt = self.prompt_generator.generate_openai_prompt(bug)
        if get_token_count(full_prompt, model="gpt-4o") <= available_tokens:
            return self.invoke_structured(full_prompt, OpenAILocalizerResponse)
        
        # Check if we need chunking
        code_files_tokens = get_token_count("\n\n".join(bug.code_files), "gpt-4o")
        max_chunk_tokens = available_tokens // 4

        if code_files_tokens <= max_chunk_tokens:
            return self.invoke_structured(full_prompt, OpenAILocalizerResponse)
        
        # Process chunks
        chunks = chunk_code_files(bug.code_files, max_chunk_tokens, "gpt-4o")
        chunk_responses = []
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = self.prompt_generator.generate_openai_prompt(bug, chunk)
            
            if get_token_count(chunk_prompt, model="gpt-4o") > available_tokens:
                continue
            
            try:
                chunk_responses.append(self.invoke_structured(chunk_prompt, OpenAILocalizerResponse))
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
        
        # Return results
        if len(chunk_responses) == 1:
            return chunk_responses[0]
        elif len(chunk_responses) > 1:
            return self._aggregate_responses(bug, chunk_responses)
        else:
            return OpenAILocalizerResponse(candidate_files=[])

    def _aggregate_responses(self, bug, chunk_responses):
        response_texts = [
            f"Files: {getattr(response, 'files', 'N/A')}\nReasoning: {getattr(response, 'reasoning', 'N/A')}"
            for response in chunk_responses
        ]
        
        aggregation_prompt = self.prompt_generator.generate_chunk_aggregation_prompt(bug, response_texts)
        
        try:
            return self.invoke_structured(aggregation_prompt, OpenAILocalizerResponse)
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return chunk_responses[0]

    def cleanup(self):
        if self.unsloth_model is not None:
            del self.unsloth_model
            del self.unsloth_tokenizer
            
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass
