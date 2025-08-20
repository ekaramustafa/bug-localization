class PromptGenerator:

    def generate_openai_prompt(bug):
        prompt = f"""
        You are a bug localization expert. Given a bug report and a list of Python files in a repository, 
        your task is to identify which files are most likely related to the reported bug.

        Instructions:
        - Analyze the bug description.
        - Consider the file names and their potential relevance.
        - Return a ranked list of Python files based on the probability that they contain the bug.

        Bug Report:
        {bug.to_string()}
        """

        return prompt
    
    