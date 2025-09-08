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


    def _get_hierarchical_files(self, bug, code_files):
        if not code_files:
            return []
        
        # Group files by their directory structure
        file_tree = {}
        for file_path in code_files:
            parts = file_path.split(os.sep)
            current = file_tree
            for part in parts[:-1]:  # All except filename
                if part not in current:
                    current[part] = {}
                current = current[part]
            # Add filename to the directory
            if '_files' not in current:
                current['_files'] = []
            current['_files'].append(file_path)
        
        # Start exploration from root
        selected_files = []
        self._explore_tree(bug, file_tree, selected_files, "")
        
        return selected_files

    def _explore_tree(self, bug, tree, selected_files, current_path, max_depth=5):
        if max_depth <= 0:
            return
        
        # Get directories and files at current level
        directories = [k for k in tree.keys() if k != '_files']
        files = tree.get('_files', [])
        
        if not directories and not files:
            return
        
        # Present options to LLM
        options = []
        if directories:
            options.extend([f"DIR: {d}" for d in directories[:10]])
        if files:
            file_names = [os.path.basename(f) for f in files[:20]]
            options.extend([f"FILE: {name}" for name in file_names])
        
        prompt = f"""
Bug: {bug.bug_report[:1000]}

Current path: {current_path or 'root'}
Available options:
{chr(10).join(options)}

Select ONE of:
1. Directory name to explore (just the name, no "DIR:" prefix)
2. Comma-separated file names to add as candidates (just names, no "FILE:" prefix)  
3. "STOP" if no relevant options

Response:"""
        
        response = self._generate_text(prompt).strip().upper()
        
        if response == "STOP":
            return
        
        if "," in response:
            # Multiple files selected
            selected_names = [f.strip() for f in response.split(",")]
            for file_path in files:
                if os.path.basename(file_path).upper() in [n.upper() for n in selected_names]:
                    selected_files.append(file_path)
        else:
            # Single directory or file
            response_lower = response.lower()
            
            # Check directories
            for d in directories:
                if d.lower() == response_lower:
                    next_path = os.path.join(current_path, d) if current_path else d
                    self._explore_tree(bug, tree[d], selected_files, next_path, max_depth - 1)
                    return
            
            # Check files
            for file_path in files:
                if os.path.basename(file_path).lower() == response_lower:
                    selected_files.append(file_path)
                    return

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
