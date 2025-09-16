import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import sys
from typing import List, Dict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from method.openrouter_localizer import OpenRouterLocalizer, DirectorySelectionResponse
from method.models import OpenAILocalizerResponse
from dataset.models import BugInstance


class TestOpenRouterHierarchical(unittest.TestCase):
    """Tests for hierarchical file selection functionality in OpenRouterLocalizer"""
    
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
        
        # Create complex directory structure for testing
        self.complex_code_files = [
            "src/auth/login.py",
            "src/auth/register.py",
            "src/auth/middleware.py",
            "src/auth/utils.py",
            "src/data/models/user.py",
            "src/data/models/session.py",
            "src/data/repositories/user_repo.py",
            "src/data/repositories/session_repo.py",
            "src/api/controllers/auth_controller.py",
            "src/api/controllers/user_controller.py",
            "src/api/routes/auth_routes.py",
            "src/api/routes/user_routes.py",
            "src/utils/validation.py",
            "src/utils/encryption.py",
            "src/utils/logging.py",
            "tests/unit/test_auth.py",
            "tests/unit/test_user.py",
            "tests/integration/test_auth_flow.py",
            "config/settings.py",
            "config/database.py",
            "requirements.txt",
            "README.md"
        ]
        
        # Create sample bug for hierarchical testing
        self.hierarchical_bug = BugInstance(
            repo="complex-repo",
            instance_id="hier-001",
            base_commit="xyz123",
            patch="diff --git a/src/auth/login.py b/src/auth/login.py\n...",
            bug_report="Users are unable to authenticate properly. The system returns authentication errors even with valid credentials.",
            hints_text="Check login logic. Verify password validation. Review session management.",
            ground_truths=["src/auth/login.py", "src/utils/validation.py"],
            code_files=self.complex_code_files
        )
    
    def tearDown(self):
        """Clean up after each test method"""
        self.env_patcher.stop()
        self.openai_patcher.stop()
        self.dotenv_patcher.stop()
    
    def test_get_hierarchical_files_empty_list(self):
        """Test hierarchical file selection with empty file list"""
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, [])
        
        self.assertEqual(result, [])
    
    def test_get_hierarchical_files_single_level_selection(self):
        """Test hierarchical file selection with single directory level"""
        # Mock API responses for directory selection
        directory_response = json.dumps({
            "selected_directory": "src/auth",
            "selected_files": []
        })
        
        file_response = json.dumps({
            "selected_directory": None,
            "selected_files": ["src/auth/login.py", "src/auth/utils.py"]
        })
        
        # Mock API calls in sequence
        mock_completion_1 = Mock()
        mock_completion_1.choices = [Mock()]
        mock_completion_1.choices[0].message.content = directory_response
        mock_completion_1.usage = Mock()
        mock_completion_1.usage.total_tokens = 50
        
        mock_completion_2 = Mock()
        mock_completion_2.choices = [Mock()]
        mock_completion_2.choices[0].message.content = file_response
        mock_completion_2.usage = Mock()
        mock_completion_2.usage.total_tokens = 75
        
        self.mock_client.chat.completions.create.side_effect = [mock_completion_1, mock_completion_2]
        
        # Test hierarchical selection
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Verify API was called twice (directory selection + file selection)
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)
        
        # Verify result contains selected files
        self.assertIn("src/auth/login.py", result)
        self.assertIn("src/auth/utils.py", result)
    
    def test_get_hierarchical_files_multi_level_selection(self):
        """Test hierarchical file selection with multiple directory levels"""
        # Mock API responses for multi-level navigation
        responses = [
            json.dumps({"selected_directory": "src", "selected_files": []}),  # Select src directory
            json.dumps({"selected_directory": "src/auth", "selected_files": []}),  # Select auth subdirectory
            json.dumps({"selected_directory": None, "selected_files": ["src/auth/login.py"]})  # Select final files
        ]
        
        # Mock API calls
        mock_completions = []
        for i, response in enumerate(responses):
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = response
            mock_completion.usage = Mock()
            mock_completion.usage.total_tokens = 50 + i * 10  # Different token counts
            mock_completions.append(mock_completion)
        
        self.mock_client.chat.completions.create.side_effect = mock_completions
        
        # Test hierarchical selection
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Verify multiple API calls were made
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 3)
        
        # Verify result
        self.assertIn("src/auth/login.py", result)
    
    def test_get_hierarchical_files_direct_file_selection(self):
        """Test hierarchical file selection with direct file selection (no directory navigation)"""
        # Mock API response for direct file selection
        file_response = json.dumps({
            "selected_directory": None,
            "selected_files": ["src/auth/login.py", "src/utils/validation.py", "config/settings.py"]
        })
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = file_response
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 100
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test hierarchical selection
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Verify single API call
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 1)
        
        # Verify all selected files are in result
        expected_files = ["src/auth/login.py", "src/utils/validation.py", "config/settings.py"]
        for file_path in expected_files:
            self.assertIn(file_path, result)
    
    def test_get_hierarchical_files_mixed_selection(self):
        """Test hierarchical file selection with mixed directory and file selection"""
        # Mock API response selecting both directory and specific files
        mixed_response = json.dumps({
            "selected_directory": "src/auth",
            "selected_files": ["config/settings.py", "src/utils/validation.py"]
        })
        
        # Mock follow-up response for the selected directory
        directory_files_response = json.dumps({
            "selected_directory": None,
            "selected_files": ["src/auth/login.py", "src/auth/middleware.py"]
        })
        
        mock_completion_1 = Mock()
        mock_completion_1.choices = [Mock()]
        mock_completion_1.choices[0].message.content = mixed_response
        mock_completion_1.usage = Mock()
        mock_completion_1.usage.total_tokens = 80
        
        mock_completion_2 = Mock()
        mock_completion_2.choices = [Mock()]
        mock_completion_2.choices[0].message.content = directory_files_response
        mock_completion_2.usage = Mock()
        mock_completion_2.usage.total_tokens = 90
        
        self.mock_client.chat.completions.create.side_effect = [mock_completion_1, mock_completion_2]
        
        # Test hierarchical selection
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Verify both API calls were made
        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)
        
        # Verify all selected files are in result
        expected_files = ["config/settings.py", "src/utils/validation.py", "src/auth/login.py", "src/auth/middleware.py"]
        for file_path in expected_files:
            self.assertIn(file_path, result)
    
    def test_get_hierarchical_files_max_depth_limit(self):
        """Test hierarchical file selection respects maximum depth limit"""
        # Create very deep directory structure
        deep_files = [f"level1/level2/level3/level4/level5/level6/file{i}.py" for i in range(5)]
        
        # Mock API responses that keep going deeper
        deep_responses = [
            json.dumps({"selected_directory": "level1", "selected_files": []}),
            json.dumps({"selected_directory": "level1/level2", "selected_files": []}),
            json.dumps({"selected_directory": "level1/level2/level3", "selected_files": []}),
            json.dumps({"selected_directory": "level1/level2/level3/level4", "selected_files": []}),
            json.dumps({"selected_directory": "level1/level2/level3/level4/level5", "selected_files": []}),
            # Should stop here due to max depth
            json.dumps({"selected_directory": None, "selected_files": ["level1/level2/level3/level4/level5/level6/file0.py"]})
        ]
        
        mock_completions = []
        for i, response in enumerate(deep_responses):
            mock_completion = Mock()
            mock_completion.choices = [Mock()]
            mock_completion.choices[0].message.content = response
            mock_completion.usage = Mock()
            mock_completion.usage.total_tokens = 60 + i * 5
            mock_completions.append(mock_completion)
        
        self.mock_client.chat.completions.create.side_effect = mock_completions
        
        # Test with deep directory structure
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, deep_files)
        
        # Verify depth limit was respected (should not make more than max_depth calls)
        self.assertLessEqual(self.mock_client.chat.completions.create.call_count, 6)
    
    def test_get_hierarchical_files_api_error_handling(self):
        """Test hierarchical file selection handles API errors gracefully"""
        from openai import APIError
        
        # Mock API error during hierarchical selection
        mock_request = Mock()
        self.mock_client.chat.completions.create.side_effect = APIError(
            "API error during selection", 
            request=mock_request,
            body={"error": "api_error"}
        )
        
        # Test hierarchical selection with error
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Should return empty list on error
        self.assertEqual(result, [])
    
    def test_get_hierarchical_files_malformed_json_response(self):
        """Test hierarchical file selection handles malformed JSON responses"""
        # Mock malformed JSON response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "This is not valid JSON"
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 30
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test hierarchical selection with malformed response
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Should handle gracefully and return empty list
        self.assertEqual(result, [])
    
    def test_get_hierarchical_files_empty_response(self):
        """Test hierarchical file selection handles empty API responses"""
        # Mock empty response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = ""
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 10
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test hierarchical selection with empty response
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, self.complex_code_files)
        
        # Should handle gracefully and return empty list
        self.assertEqual(result, [])
    
    def test_explore_tree_basic_functionality(self):
        """Test the _explore_tree method basic functionality"""
        # Create a simple file tree structure
        file_tree = {
            "src": {
                "auth": {
                    "__files__": ["login.py", "register.py"]
                },
                "utils": {
                    "__files__": ["validation.py", "encryption.py"]
                }
            },
            "tests": {
                "__files__": ["test_auth.py"]
            }
        }
        
        # Mock API response for tree exploration
        response = json.dumps({
            "selected_directory": "src/auth",
            "selected_files": ["tests/test_auth.py"]
        })
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = response
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test tree exploration
        selected_files = []
        self.localizer._explore_tree(self.hierarchical_bug, file_tree, selected_files, "", max_depth=3)
        
        # Verify files were selected
        self.assertIn("tests/test_auth.py", selected_files)
    
    def test_explore_tree_max_depth_reached(self):
        """Test _explore_tree stops at maximum depth"""
        # Create nested tree structure
        file_tree = {
            "level1": {
                "level2": {
                    "level3": {
                        "__files__": ["deep_file.py"]
                    }
                }
            }
        }
        
        # Test with max_depth=2 (should not reach level3)
        selected_files = []
        
        # Mock API response
        response = json.dumps({
            "selected_directory": "level1/level2/level3",
            "selected_files": []
        })
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = response
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Should not explore beyond max_depth
        self.localizer._explore_tree(self.hierarchical_bug, file_tree, selected_files, "", max_depth=2)
        
        # Verify depth limit was respected
        self.assertEqual(len(selected_files), 0)  # Should not have selected deep files
    
    def test_hierarchical_selection_with_token_management(self):
        """Test hierarchical selection considers token limits"""
        # Create a large number of files to test token management
        large_file_list = [f"src/module_{i}/file_{j}.py" for i in range(50) for j in range(20)]  # 1000 files
        
        # Mock API response that selects a reasonable subset
        response = json.dumps({
            "selected_directory": None,
            "selected_files": large_file_list[:10]  # Select first 10 files
        })
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = response
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 120
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test hierarchical selection with large file list
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, large_file_list)
        
        # Verify reasonable number of files selected (not all 1000)
        self.assertLessEqual(len(result), 50)  # Should be manageable number
        self.assertGreater(len(result), 0)  # Should select some files
    
    def test_hierarchical_selection_file_tree_building(self):
        """Test that file tree is built correctly from file paths"""
        # Test files with various path structures
        test_files = [
            "src/auth/login.py",
            "src/auth/register.py",
            "src/data/models/user.py",
            "tests/unit/test_auth.py",
            "config.py",
            "README.md"
        ]
        
        # Mock API response
        response = json.dumps({
            "selected_directory": "src",
            "selected_files": ["config.py"]
        })
        
        follow_up_response = json.dumps({
            "selected_directory": None,
            "selected_files": ["src/auth/login.py"]
        })
        
        mock_completion_1 = Mock()
        mock_completion_1.choices = [Mock()]
        mock_completion_1.choices[0].message.content = response
        mock_completion_1.usage = Mock()
        mock_completion_1.usage.total_tokens = 70
        
        mock_completion_2 = Mock()
        mock_completion_2.choices = [Mock()]
        mock_completion_2.choices[0].message.content = follow_up_response
        mock_completion_2.usage = Mock()
        mock_completion_2.usage.total_tokens = 85
        
        self.mock_client.chat.completions.create.side_effect = [mock_completion_1, mock_completion_2]
        
        # Test hierarchical selection
        result = self.localizer._get_hierarchical_files(self.hierarchical_bug, test_files)
        
        # Verify both direct files and directory exploration results
        self.assertIn("config.py", result)
        self.assertIn("src/auth/login.py", result)


if __name__ == '__main__':
    unittest.main()