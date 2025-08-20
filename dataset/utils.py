import logging
import sys
from datetime import datetime
import tiktoken  # You'll need to install: pip install tiktoken

def setup_logging(level=logging.INFO, log_file=None):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging initialized successfully")
    return root_logger

def get_logger(name):
    return logging.getLogger(name)



import requests

def get_code_files(repo, commit_hash, extensions, token=None):
    url = f"https://api.github.com/repos/{repo}/git/trees/{commit_hash}?recursive=1"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    tree = r.json()["tree"]
    py_files = [item["path"] for item in tree if item["type"] == "blob" and item["path"].endswith(extensions)]
    return py_files

def get_token_count(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken for the specified model.
    
    Args:
        text: The text to count tokens for
        model: The model name to get the appropriate encoder for
    
    Returns:
        Number of tokens
    """
    try:
        # Map model names to encoders
        encoder_map = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "o200k_base", 
            "gpt-3.5-turbo": "cl100k_base",
        }
        
        encoder_name = encoder_map.get(model, "cl100k_base")
        encoder = tiktoken.get_encoding(encoder_name)
        return len(encoder.encode(text))
    except Exception as e:
        logging.warning(f"Failed to count tokens: {e}")
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def calculate_dataset_token_stats(bug_instances, model="gpt-4o"):
    """
    Calculate token statistics for a dataset of bug instances.
    
    Args:
        bug_instances: List of BugInstance objects
        model: Model name for token counting
    
    Returns:
        Dictionary with token statistics
    """
    token_counts = []
    
    for bug in bug_instances:
        # Count tokens in the bug report
        bug_report_tokens = get_token_count(bug.bug_report, model)
        
        # Count tokens in hints
        hints_tokens = get_token_count(bug.hints_text, model)
        
        # Count tokens in code file list (as it would appear in prompt)
        code_files_text = ', '.join(bug.code_files)
        code_files_tokens = get_token_count(code_files_text, model)
        
        # Total tokens for this instance (approximate prompt size)
        total_tokens = bug_report_tokens + hints_tokens + code_files_tokens
        token_counts.append({
            'instance_id': bug.instance_id,
            'bug_report_tokens': bug_report_tokens,
            'hints_tokens': hints_tokens,
            'code_files_tokens': code_files_tokens,
            'total_tokens': total_tokens,
            'num_code_files': len(bug.code_files)
        })
    
    # Calculate statistics
    total_tokens_list = [tc['total_tokens'] for tc in token_counts]
    
    stats = {
        'total_instances': len(token_counts),
        'mean_tokens': sum(total_tokens_list) / len(total_tokens_list),
        'min_tokens': min(total_tokens_list),
        'max_tokens': max(total_tokens_list),
        'total_tokens': sum(total_tokens_list),
        'detailed_counts': token_counts
    }
    
    return stats