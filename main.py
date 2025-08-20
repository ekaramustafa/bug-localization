from dataset.swebench import SWEBench
from method.openai_localizer import OpenAILocalizer
from dataset.utils import setup_logging, get_logger
import logging
from method.evaluate import Evaluator

# Initialize logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def main():
    logger.info("Starting...")
    
    try:
        instance = SWEBench()
        localizer = OpenAILocalizer()
        bug_instances = instance.get_bug_instances()
        logger.info(f"Retrieved {len(bug_instances)} bug instances")

        # Get dataset token statistics
        token_stats = instance.get_token_statistics()
        print(token_stats)
        return 
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
        
        # Log final usage summary
        usage_summary = localizer.get_usage_summary()
        logger.info(f"Final API Usage Summary: {usage_summary}")
        
        evaluator = Evaluator()
        results = evaluator.evaluate(responses)
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


