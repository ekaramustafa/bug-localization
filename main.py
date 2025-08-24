from dataset.swebench import SWEBench
from dataset.beetlebox import BeetleBox
from method.openai_localizer import OpenAILocalizer
from method.openai_free_localizer import OpenAIFreeLocalizer
from dataset.utils import setup_logging, get_logger
import logging
from method.evaluate import Evaluator

# Initialize logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def main():
    logger.info("Starting...")
    
    try:
        # instance = SWEBench()
        instance = BeetleBox()
        # localizer = OpenAILocalizer()
        localizer = OpenAIFreeLocalizer()
        
        # Example 1: Get all instances (default behavior)
        # bug_instances = instance.get_bug_instances()
        
        # Example 2: Get first 10 instances
        # bug_instances = instance.get_bug_instances(sample_size=10)
        
        # Example 3: Get 10 random instances with reproducible seed
        bug_instances = instance.get_bug_instances(sample_size=100, random_sample=True, random_seed=42)
        
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
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


