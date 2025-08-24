from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from method.openai_localizer import OpenAILocalizer
from method.openai_free_localizer import OpenAIFreeLocalizer
from method.huggingface_localizer import HuggingFaceLocalizer
from dataset.utils import setup_logging, get_logger
import logging
from method.evaluate import Evaluator
import argparse

# Initialize logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Bug Localization Tool')
    parser.add_argument('--method', choices=['openai', 'openai-free', 'huggingface'], 
                       default='openai-free', help='Localization method to use')
    parser.add_argument('--dataset', choices=['swebench', 'beetlebox'], 
                       default='beetlebox', help='Dataset to use')
    parser.add_argument('--model', default='gpt-oss', 
                       help='Model to use (for HuggingFace: gpt-oss, gpt-oss-120b, etc.)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], 
                       default='auto', help='Device for local inference (HuggingFace only)')
    parser.add_argument('--sample-size', type=int, default=100, 
                       help='Number of bug instances to process')
    
    args = parser.parse_args()
    
    logger.info("Starting...")
    logger.info(f"Method: {args.method}, Dataset: {args.dataset}, Model: {args.model}")
    
    try:
        # Initialize dataset
        if args.dataset == 'swebench':
            instance = SWEBench()
        else:
            instance = BeetleBox()
        
        # Initialize localizer based on method
        if args.method == 'openai':
            localizer = OpenAILocalizer()
        elif args.method == 'openai-free':
            localizer = OpenAIFreeLocalizer(model=args.model)
        elif args.method == 'huggingface':
            device = None if args.device == 'auto' else args.device
            localizer = HuggingFaceLocalizer(
                model=args.model,
                device=device
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # Get bug instances based on sample size argument
        bug_instances = instance.get_bug_instances(sample_size=args.sample_size, random_sample=True, random_seed=42)
        
        logger.info(f"Retrieved {len(bug_instances)} bug instances")
        # Get dataset token statistics
        token_stats = instance.get_token_statistics()
        logger.info(f"Token statistics: {token_stats}")

        logger.info(f"Total repo: {len(instance.repos)}")
        responses = {}
        for i, bug in enumerate(bug_instances): 
            patch = bug.patch
            ground_truths = bug.ground_truths
            # logger.info(f"Ground truths: {ground_truths}")
            # logger.info(f"First 3 py files: {bug.code_files[:3]}")
            # logger.info(f"Total py files: {len(bug.code_files)}")
        
            response = localizer.localize(bug)
            responses[bug.instance_id] = {'bug': bug, 'response': response}
            logger.info(f"Response: {response}")
        
        evaluator = Evaluator()
        results = evaluator.evaluate(responses)
        logger.info(f"Results: {results}")
        
        # Cleanup if using HuggingFace localizer
        if hasattr(localizer, 'cleanup'):
            localizer.cleanup()
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        # Cleanup on error if using HuggingFace localizer
        if 'localizer' in locals() and hasattr(localizer, 'cleanup'):
            localizer.cleanup()
        raise


if __name__ == "__main__":
    main()


