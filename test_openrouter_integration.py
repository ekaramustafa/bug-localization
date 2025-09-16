import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import sys
from typing import List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from method.openrouter_localizer import OpenRouterLocalizer
from method.models import OpenAILocalizerResponse
from dataset.models import BugInstance


class TestOpenRouterIntegration(unittest.TestCase):
    """Integration tests for OpenRouterLocalizer with sample bug instances"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OPENROUTER_API_KEY': 'test-api-key-12345'
        })
        self.env_patcher.start()
        
        # Mock the OpenAI client to avoid actual API calls
        self.openai_patcher = patch('method.openrouter_localizer.OpenAI')
        self.mock_openai_class = self.openai_patcher.start()
        self.mock_client = Mock()
        self.mock_openai_class.return_value = self.mock_client
        
        # Mock dotenv load
        self.dotenv_patcher = patch('method.openrouter_localizer.load_dotenv')
        self.dotenv_patcher.start()
        
        # Create test instance
        self.localizer = OpenRouterLocalizer(model="qwen-coder-32b")
        
        # Create sample bug instance
        self.sample_bug = BugInstance(
            repo="test-repo",
            instance_id="test-001",
            base_commit="abc123",
            patch="diff --git a/src/auth/login.py b/src/auth/login.py\n...",
            bug_report="Users cannot log in with valid credentials. The login function returns false even with correct username and password.",
            hints_text="Check authentication logic. Verify password hashing.",
            ground_truths=["src/auth/login.py", "src/utils/password.py"],
            code_files=[
                "src/auth/login.py",
                "src/auth/register.py", 
                "src/utils/password.py",
                "src/utils/validation.py",
                "src/models/user.py",
                "src/controllers/auth_controller.py",
                "tests/test_auth.py",
                "config/settings.py"
            ]
        )
    
    def tearDown(self):
        """Clean up after each test method"""
        self.env_patcher.stop()
        self.openai_patcher.stop()
        self.dotenv_patcher.stop()
    
    def test_localize_with_sample_bug_instance_success(self):
        """Test complete localization workflow with sample bug instance"""
        # Mock successful API response for localization
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "candidate_files": ["src/auth/login.py", "src/utils/password.py"]
        })
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 150
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test localization prompt"
            
            # Execute localization
            result = self.localizer.localize(self.sample_bug)
            
            # Verify result
            self.assertIsInstance(result, OpenAILocalizerResponse)
            self.assertEqual(result.candidate_files, ["src/auth/login.py", "src/utils/password.py"])
            
            # Verify API was called
            self.mock_client.chat.completions.create.assert_called()
            
            # Verify prompt generation was called
            mock_prompt_gen.generate_openai_prompt.assert_called_once()
    
    def test_localize_with_token_limit_exceeded(self):
        """Test localization when token limit is exceeded and hierarchical selection is used"""
        # Create a bug with many files to trigger hierarchical selection
        large_bug = BugInstance(
            repo="large-repo",
            instance_id="large-001",
            base_commit="def456",
            patch="diff --git a/src/data/processor.py b/src/data/processor.py\n...",
            bug_report="The data processing pipeline is very slow when handling large datasets.",
            hints_text="Check database queries. Review caching logic.",
            ground_truths=["src/data/processor.py"],
            code_files=[f"src/module_{i}/file_{j}.py" for i in range(10) for j in range(20)]  # 200 files
        )
        
        # Mock hierarchical selection responses
        directory_selection_response = json.dumps({
            "selected_directory": "src/data",
            "selected_files": []
        })
        
        final_localization_response = json.dumps({
            "candidate_files": ["src/data/processor.py", "src/data/cache.py"]
        })
        
        # Mock API responses in sequence
        mock_completion_1 = Mock()
        mock_completion_1.choices = [Mock()]
        mock_completion_1.choices[0].message.content = directory_selection_response
        mock_completion_1.usage = Mock()
        mock_completion_1.usage.total_tokens = 100
        
        mock_completion_2 = Mock()
        mock_completion_2.choices = [Mock()]
        mock_completion_2.choices[0].message.content = final_localization_response
        mock_completion_2.usage = Mock()
        mock_completion_2.usage.total_tokens = 200
        
        self.mock_client.chat.completions.create.side_effect = [mock_completion_1, mock_completion_2]
        
        # Mock prompt generator and hierarchical selection
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            with patch.object(self.localizer, '_get_hierarchical_files') as mock_hierarchical:
                # Mock token counting to force hierarchical selection
                with patch('dataset.utils.get_token_count') as mock_token_count:
                    mock_token_count.return_value = 10000  # Force hierarchical selection
                    mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
                    mock_hierarchical.return_value = ["src/data/processor.py", "src/data/cache.py"]
                    
                    # Execute localization
                    result = self.localizer.localize(large_bug)
                    
                    # Verify hierarchical selection was used
                    mock_hierarchical.assert_called_once()
                    
                    # Verify result
                    self.assertIsInstance(result, OpenAILocalizerResponse)
                    self.assertEqual(result.candidate_files, ["src/data/processor.py", "src/data/cache.py"])
    
    def test_localize_with_api_error_handling(self):
        """Test localization with API error handling"""
        from openai import APIError
        
        # Create mock response and request objects for the exception
        mock_request = Mock()
        self.mock_client.chat.completions.create.side_effect = APIError(
            "API temporarily unavailable", 
            request=mock_request,
            body={"error": "api_unavailable"}
        )
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
            
            # Execute localization and expect it to handle error gracefully
            with self.assertRaises(ValueError):
                self.localizer.localize(self.sample_bug)
    
    def test_localize_with_malformed_json_response(self):
        """Test localization with malformed JSON response"""
        # Mock API response with malformed JSON
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "This is not valid JSON: {candidate_files: [file1.py]"
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 50
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
            
            # Mock fallback parsing to return empty response
            with patch.object(self.localizer, '_fallback_structured_parsing') as mock_fallback:
                mock_fallback.return_value = OpenAILocalizerResponse(candidate_files=[])
                
                # Execute localization
                result = self.localizer.localize(self.sample_bug)
                
                # Verify fallback was used
                mock_fallback.assert_called_once()
                
                # Verify empty result
                self.assertIsInstance(result, OpenAILocalizerResponse)
                self.assertEqual(result.candidate_files, [])
    
    def test_localize_with_empty_code_files(self):
        """Test localization with empty code files list"""
        empty_bug = BugInstance(
            repo="empty-repo",
            instance_id="empty-001",
            base_commit="ghi789",
            patch="",
            bug_report="Test description",
            hints_text="",
            ground_truths=[],
            code_files=[]
        )
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
            
            # Mock API response
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = json.dumps({"candidate_files": []})
            mock_completion.usage = Mock()
            mock_completion.usage.total_tokens = 20
            
            self.mock_client.chat.completions.create.return_value = mock_completion
            
            # Execute localization
            result = self.localizer.localize(empty_bug)
            
            # Verify result
            self.assertIsInstance(result, OpenAILocalizerResponse)
            self.assertEqual(result.candidate_files, [])
    
    def test_localize_workflow_logging_and_statistics(self):
        """Test that localization workflow properly logs and tracks statistics"""
        # Mock successful API response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "candidate_files": ["src/auth/login.py"]
        })
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_completion.usage.total_tokens = 150
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
            
            # Execute localization
            result = self.localizer.localize(self.sample_bug)
            
            # Verify statistics tracking
            self.assertEqual(self.localizer._api_call_count, 1)
            self.assertEqual(self.localizer._total_tokens_used, 150)
            self.assertEqual(self.localizer._successful_requests, 1)
            self.assertEqual(self.localizer._failed_requests, 0)
    
    def test_end_to_end_workflow_with_cleanup(self):
        """Test complete end-to-end workflow including cleanup"""
        # Mock successful API response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "candidate_files": ["src/auth/login.py", "src/utils/password.py"]
        })
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 200
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
            
            # Execute complete workflow
            result = self.localizer.localize(self.sample_bug)
            
            # Verify result
            self.assertIsInstance(result, OpenAILocalizerResponse)
            self.assertEqual(len(result.candidate_files), 2)
            
            # Test cleanup
            initial_api_calls = self.localizer._api_call_count
            self.localizer.cleanup()
            
            # Verify cleanup was successful
            self.assertIsNone(self.localizer.client)
            self.assertEqual(len(self.localizer.model_mapping), 0)
    
    def test_multiple_localizations_statistics_accumulation(self):
        """Test that statistics accumulate correctly across multiple localizations"""
        # Mock successful API responses
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "candidate_files": ["test.py"]
        })
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 100
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Mock prompt generator
        with patch.object(self.localizer, 'prompt_generator') as mock_prompt_gen:
            mock_prompt_gen.generate_openai_prompt.return_value = "Test prompt"
            
            # Execute multiple localizations
            for i in range(3):
                result = self.localizer.localize(self.sample_bug)
                self.assertIsInstance(result, OpenAILocalizerResponse)
            
            # Verify accumulated statistics
            self.assertEqual(self.localizer._api_call_count, 3)
            self.assertEqual(self.localizer._total_tokens_used, 300)
            self.assertEqual(self.localizer._successful_requests, 3)
            self.assertEqual(self.localizer._failed_requests, 0)


if __name__ == '__main__':
    unittest.main()