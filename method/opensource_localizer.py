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
from dataset.utils import get_token_count, get_logger

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

    def _scan_directory(self, path):
        directories = []
        files = []
        
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    directories.append(item)
                elif os.path.isfile(item_path):
                    files.append(item)
        except:
            pass
        
        return directories, files

    def _select_path(self, bug, current_path, directories, files):
        options = []
        if directories:
            options.extend([f"DIR: {d}" for d in directories[:10]])  # Limit to avoid context overflow
        if files:
            options.extend([f"FILE: {f}" for f in files[:20]])
        
        prompt = f"""
Bug: {bug.bug_report[:1000]}

Current path: {current_path}
Available options:
{chr(10).join(options)}

Select ONE of:
1. Directory name to explore (just the name, no "DIR:" prefix)
2. Comma-separated file names to add as candidates (just names, no "FILE:" prefix)  
3. "STOP" if no relevant options

Response:"""
        
        response = self._generate_text(prompt).strip().upper()
        
        if response == "STOP":
            return None, []
        
        if "," in response:
            # Multiple files selected
            selected_files = [f.strip() for f in response.split(",")]
            return None, [f for f in selected_files if f in files]
        
        # Single directory or file
        response_lower = response.lower()
        
        for d in directories:
            if d.lower() == response_lower:
                return d, []
        
        for f in files:
            if f.lower() == response_lower:
                return None, [f]
        
        return None, []

    def _explore_hierarchy(self, bug, base_path, current_path="", max_depth=5):
        if max_depth <= 0:
            return []
        
        full_path = os.path.join(base_path, current_path) if current_path else base_path
        candidate_files = []
        
        directories, files = self._scan_directory(full_path)
        
        if not directories and not files:
            return []
        
        selected_dir, selected_files = self._select_path(bug, current_path or ".", directories, files)
        
        # Add selected files to candidates
        for file in selected_files:
            file_path = os.path.join(current_path, file) if current_path else file
            candidate_files.append(file_path)
        
        # Explore selected directory
        if selected_dir:
            next_path = os.path.join(current_path, selected_dir) if current_path else selected_dir
            candidate_files.extend(self._explore_hierarchy(bug, base_path, next_path, max_depth - 1))
        
        return candidate_files

    def _get_hierarchical_files(self, bug, code_files):
        # Extract base path from first code file
        if not code_files:
            return []
        
        # Find common base directory
        base_path = os.path.commonpath([os.path.dirname(f) for f in code_files])
        
        # Explore hierarchy starting from base path
        relative_files = self._explore_hierarchy(bug, base_path)
        
        # Convert back to full paths
        full_paths = [os.path.join(base_path, rf) for rf in relative_files]
        
        # Filter to only include files that exist in original code_files
        return [f for f in full_paths if f in code_files]

    def localize(self, bug):
        available_tokens = self.max_seq_length - self.max_new_tokens - 250
        
        # Try direct processing first
        full_prompt = self.prompt_generator.generate_openai_prompt(bug)
        if get_token_count(full_prompt, model="gpt-4o") <= available_tokens:
            return self.invoke_structured(full_prompt, OpenAILocalizerResponse)
        
        # Use hierarchical approach to select relevant files
        selected_files = self._get_hierarchical_files(bug, bug.code_files)
        
        if not selected_files:
            return OpenAILocalizerResponse(candidate_files=[])
        
        # Create prompt with selected files
        selected_prompt = self.prompt_generator.generate_openai_prompt(bug, selected_files)
        
        # If still too large, take first batch that fits
        if get_token_count(selected_prompt, model="gpt-4o") > available_tokens:
            for i in range(len(selected_files), 0, -1):
                subset_prompt = self.prompt_generator.generate_openai_prompt(bug, selected_files[:i])
                if get_token_count(subset_prompt, model="gpt-4o") <= available_tokens:
                    return self.invoke_structured(subset_prompt, OpenAILocalizerResponse)
            return OpenAILocalizerResponse(candidate_files=[])
        
        return self.invoke_structured(selected_prompt, OpenAILocalizerResponse)


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
