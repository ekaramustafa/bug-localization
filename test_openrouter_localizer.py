import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from method.openrouter_localizer import OpenRouterLocalizer, DirectorySelectionResponse
from method.models import OpenAILocalizerResponse
from dataset.models import BugInstance


class TestOpenRouterLocalizer(unittest.TestCase):
    """Comprehensive unit tests for OpenRouterLocalizer"""
    
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
        self.localizer = OpenRouterLocalizer(model="qwen-coder-32b", max_tokens=2048, temperature=0.5)
    
    def tearDown(self):
        """Clean up after each test method"""
        self.env_patcher.stop()
        self.openai_patcher.stop()
        self.dotenv_patcher.stop()
    
    def test_initialization_with_valid_api_key(self):
        """Test successful initialization with valid API key"""
        self.assertEqual(self.localizer.model, "qwen-coder-32b")
        self.assertEqual(self.localizer.max_tokens, 2048)
        self.assertEqual(self.localizer.temperature, 0.5)
        self.assertEqual(self.localizer.api_key, "test-api-key-12345")
        
        # Verify OpenAI client was initialized with correct parameters
        self.mock_openai_class.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-api-key-12345"
        )
    
    def test_initialization_without_api_key(self):
        """Test initialization fails without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                OpenRouterLocalizer()
            self.assertIn("OPENROUTER_API_KEY environment variable is required", str(context.exception))
    
    def test_model_validation_friendly_name(self):
        """Test model validation with friendly names"""
        # Test valid friendly name
        validated = self.localizer._validate_and_set_model("qwen-coder-32b")
        self.assertEqual(validated, "qwen-coder-32b")
        
        # Test another valid friendly name
        validated = self.localizer._validate_and_set_model("gpt-oss-20b")
        self.assertEqual(validated, "gpt-oss-20b")
    
    def test_model_validation_full_id(self):
        """Test model validation with full OpenRouter IDs"""
        # Test valid full ID that exists in mapping
        validated = self.localizer._validate_and_set_model("qwen/qwen-2.5-coder-32b-instruct")
        self.assertEqual(validated, "qwen-coder-32b")
        
        # Test unknown full ID
        validated = self.localizer._validate_and_set_model("unknown/model-id")
        self.assertEqual(validated, "unknown/model-id")
    
    def test_model_validation_invalid_model(self):
        """Test model validation with invalid models"""
        # Test invalid model falls back to default
        validated = self.localizer._validate_and_set_model("invalid-model")
        self.assertEqual(validated, "qwen-coder-32b")  # default model
        
        # Test empty model falls back to default
        validated = self.localizer._validate_and_set_model("")
        self.assertEqual(validated, "qwen-coder-32b")
        
        # Test None model falls back to default
        validated = self.localizer._validate_and_set_model(None)
        self.assertEqual(validated, "qwen-coder-32b")
    
    def test_model_validation_partial_match(self):
        """Test model validation with partial matches"""
        # Test partial match (case insensitive)
        validated = self.localizer._validate_and_set_model("qwen")
        self.assertIn("qwen", validated.lower())
        
        # Test partial match with different case
        validated = self.localizer._validate_and_set_model("CODER")
        self.assertIn("coder", validated.lower())
    
    def test_get_model_id(self):
        """Test getting full OpenRouter model ID"""
        self.localizer.model = "qwen-coder-32b"
        model_id = self.localizer.get_model_id()
        self.assertEqual(model_id, "qwen/qwen-2.5-coder-32b-instruct")
        
        # Test with unknown model
        self.localizer.model = "unknown-model"
        model_id = self.localizer.get_model_id()
        self.assertEqual(model_id, "unknown-model")
    
    def test_get_available_models(self):
        """Test getting available models dictionary"""
        models = self.localizer.get_available_models()
        self.assertIsInstance(models, dict)
        self.assertIn("qwen-coder-32b", models)
        self.assertIn("gpt-oss-20b", models)
        self.assertEqual(models["qwen-coder-32b"], "qwen/qwen-2.5-coder-32b-instruct")
    
    def test_list_models_by_category(self):
        """Test categorizing models by type"""
        categories = self.localizer.list_models_by_category()
        self.assertIsInstance(categories, dict)
        self.assertIn("free", categories)
        self.assertIn("coding", categories)
        self.assertIn("general", categories)
        self.assertIn("specialized", categories)
        
        # Check that free models are properly categorized
        self.assertIn("gpt-oss-20b", categories["free"])
        
        # Check that coding models are properly categorized
        self.assertIn("qwen-coder-32b", categories["coding"])
    
    def test_make_api_request_success(self):
        """Test successful API request"""
        # Mock successful API response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response content"
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_completion.usage.total_tokens = 30
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test the API request
        response = self.localizer._make_api_request("Test prompt")
        
        # Verify response
        self.assertEqual(response, "Test response content")
        
        # Verify API call was made with correct parameters
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        
        self.assertEqual(call_args["model"], "qwen/qwen-2.5-coder-32b-instruct")
        self.assertEqual(call_args["messages"], [{"role": "user", "content": "Test prompt"}])
        self.assertEqual(call_args["max_tokens"], 2048)
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertIn("extra_headers", call_args)
        self.assertIn("HTTP-Referer", call_args["extra_headers"])
        self.assertIn("X-Title", call_args["extra_headers"])
    
    def test_make_api_request_structured(self):
        """Test API request with structured response format"""
        # Mock successful structured API response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = '{"candidate_files": ["file1.py", "file2.py"]}'
        mock_completion.usage = Mock()
        mock_completion.usage.total_tokens = 50
        
        self.mock_client.chat.completions.create.return_value = mock_completion
        
        # Test structured API request
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "strict": True,
                "schema": {"type": "object", "properties": {"test": {"type": "string"}}}
            }
        }
        
        response = self.localizer._make_api_request("Test prompt", structured=True, response_format=response_format)
        
        # Verify response
        self.assertEqual(response, '{"candidate_files": ["file1.py", "file2.py"]}')
        
        # Verify structured format was included
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["response_format"], response_format)
    
    def test_make_api_request_authentication_error(self):
        """Test API request with authentication error"""
        from openai import AuthenticationError
        
        # Create mock response and request objects for the exception
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request = Mock()
        
        self.mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key", response=mock_response, body={"error": "invalid_api_key"}
        )
        
        with self.assertRaises(ValueError) as context:
            self.localizer._make_api_request("Test prompt")
        
        self.assertIn("Invalid OpenRouter API key", str(context.exception))
    
    def test_make_api_request_rate_limit_error(self):
        """Test API request with rate limit error"""
        from openai import RateLimitError
        
        # Create mock response and request objects for the exception
        mock_response = Mock()
        mock_response.status_code = 429
        
        self.mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=mock_response, body={"error": "rate_limit_exceeded"}
        )
        
        with self.assertRaises(ValueError) as context:
            self.localizer._make_api_request("Test prompt")
        
        self.assertIn("Rate limit exceeded", str(context.exception))
    
    def test_make_api_request_connection_error(self):
        """Test API request with connection error"""
        from openai import APIConnectionError
        
        self.mock_client.chat.completions.create.side_effect = APIConnectionError(request=Mock())
        
        with self.assertRaises(ValueError) as context:
            self.localizer._make_api_request("Test prompt")
        
        self.assertIn("Network connection error", str(context.exception))
    
    def test_make_api_request_model_not_available(self):
        """Test API request with model not available error"""
        from openai import APIError
        
        mock_request = Mock()
        self.mock_client.chat.completions.create.side_effect = APIError(
            "404 No endpoints found for model", 
            request=mock_request,
            body={"error": "model_not_found"}
        )
        
        with self.assertRaises(ValueError) as context:
            self.localizer._make_api_request("Test prompt")
        
        self.assertIn("not currently available on OpenRouter", str(context.exception))
    
    def test_make_api_request_generic_api_error(self):
        """Test API request with generic API error"""
        from openai import APIError
        
        mock_request = Mock()
        self.mock_client.chat.completions.create.side_effect = APIError(
            "Generic API error", 
            request=mock_request,
            body={"error": "generic_error"}
        )
        
        with self.assertRaises(ValueError) as context:
            self.localizer._make_api_request("Test prompt")
        
        self.assertIn("API error", str(context.exception))
    
    def test_make_api_request_unexpected_error(self):
        """Test API request with unexpected error"""
        self.mock_client.chat.completions.create.side_effect = Exception("Unexpected error")
        
        with self.assertRaises(ValueError) as context:
            self.localizer._make_api_request("Test prompt")
        
        self.assertIn("Unexpected error", str(context.exception))
    
    def test_invoke_success(self):
        """Test successful text generation via invoke method"""
        # Mock the _make_api_request method
        with patch.object(self.localizer, '_make_api_request', return_value="Generated text response"):
            response = self.localizer.invoke("Test prompt")
            
            self.assertEqual(response, "Generated text response")
            self.localizer._make_api_request.assert_called_once_with("Test prompt")
    
    def test_invoke_error_propagation(self):
        """Test that invoke properly propagates errors from API request"""
        with patch.object(self.localizer, '_make_api_request', side_effect=ValueError("API error")):
            with self.assertRaises(ValueError):
                self.localizer.invoke("Test prompt")
    
    def test_generate_json_schema_success(self):
        """Test JSON schema generation from Pydantic model"""
        schema = self.localizer._generate_json_schema(OpenAILocalizerResponse)
        
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "json_schema")
        self.assertIn("json_schema", schema)
        self.assertIn("name", schema["json_schema"])
        self.assertIn("schema", schema["json_schema"])
        self.assertTrue(schema["json_schema"]["strict"])
        
        # Check that the schema contains expected properties
        properties = schema["json_schema"]["schema"]["properties"]
        self.assertIn("candidate_files", properties)
        self.assertEqual(properties["candidate_files"]["type"], "array")
    
    def test_generate_json_schema_fallback(self):
        """Test JSON schema generation fallback for invalid models"""
        # Create a mock model that will cause schema generation to fail
        class InvalidModel:
            pass
        
        schema = self.localizer._generate_json_schema(InvalidModel)
        
        # Should fall back to basic schema
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "json_schema")
        self.assertEqual(schema["json_schema"]["name"], "bug_localization_response")
    
    def test_invoke_structured_success(self):
        """Test successful structured response generation"""
        # Mock successful structured API call
        json_response = '{"candidate_files": ["file1.py", "file2.py"]}'
        
        with patch.object(self.localizer, '_make_api_request', return_value=json_response):
            response = self.localizer.invoke_structured("Test prompt", OpenAILocalizerResponse)
            
            self.assertIsInstance(response, OpenAILocalizerResponse)
            self.assertEqual(response.candidate_files, ["file1.py", "file2.py"])
    
    def test_invoke_structured_json_parse_error(self):
        """Test structured response with JSON parsing error"""
        # Mock API response with invalid JSON
        invalid_json = "This is not valid JSON"
        
        with patch.object(self.localizer, '_make_api_request', return_value=invalid_json):
            with patch.object(self.localizer, '_fallback_structured_parsing') as mock_fallback:
                mock_fallback.return_value = OpenAILocalizerResponse(candidate_files=[])
                
                response = self.localizer.invoke_structured("Test prompt", OpenAILocalizerResponse)
                
                # Should call fallback method
                mock_fallback.assert_called_once()
                self.assertIsInstance(response, OpenAILocalizerResponse)
    
    def test_invoke_structured_empty_response(self):
        """Test structured response with empty API response"""
        with patch.object(self.localizer, '_make_api_request', return_value=""):
            with patch.object(self.localizer, '_create_empty_response') as mock_empty:
                mock_empty.return_value = OpenAILocalizerResponse(candidate_files=[])
                
                response = self.localizer.invoke_structured("Test prompt", OpenAILocalizerResponse)
                
                mock_empty.assert_called_once_with(OpenAILocalizerResponse)
                self.assertIsInstance(response, OpenAILocalizerResponse)
    
    def test_fallback_structured_parsing_success(self):
        """Test fallback structured parsing with JSON extraction"""
        # Mock unstructured response containing JSON
        unstructured_response = 'Here is the result: {"candidate_files": ["test.py"]} and some more text'
        
        with patch.object(self.localizer, '_make_api_request', return_value=unstructured_response):
            response = self.localizer._fallback_structured_parsing("Test prompt", OpenAILocalizerResponse)
            
            self.assertIsInstance(response, OpenAILocalizerResponse)
            self.assertEqual(response.candidate_files, ["test.py"])
    
    def test_fallback_structured_parsing_no_json(self):
        """Test fallback structured parsing with no JSON found"""
        # Mock response with no JSON
        no_json_response = "This response contains no JSON at all"
        
        with patch.object(self.localizer, '_make_api_request', return_value=no_json_response):
            with patch.object(self.localizer, '_create_empty_response') as mock_empty:
                mock_empty.return_value = OpenAILocalizerResponse(candidate_files=[])
                
                response = self.localizer._fallback_structured_parsing("Test prompt", OpenAILocalizerResponse)
                
                mock_empty.assert_called_once()
    
    def test_fallback_structured_parsing_error(self):
        """Test fallback structured parsing with error"""
        with patch.object(self.localizer, '_make_api_request', side_effect=Exception("API error")):
            with patch.object(self.localizer, '_create_empty_response') as mock_empty:
                mock_empty.return_value = OpenAILocalizerResponse(candidate_files=[])
                
                response = self.localizer._fallback_structured_parsing("Test prompt", OpenAILocalizerResponse)
                
                mock_empty.assert_called_once()
    
    def test_create_empty_response_openai_localizer_response(self):
        """Test creating empty response for OpenAILocalizerResponse"""
        response = self.localizer._create_empty_response(OpenAILocalizerResponse)
        
        self.assertIsInstance(response, OpenAILocalizerResponse)
        self.assertEqual(response.candidate_files, [])
    
    def test_create_empty_response_directory_selection_response(self):
        """Test creating empty response for DirectorySelectionResponse"""
        response = self.localizer._create_empty_response(DirectorySelectionResponse)
        
        self.assertIsInstance(response, DirectorySelectionResponse)
        self.assertEqual(response.selected_files, [])
        self.assertIsNone(response.selected_directory)
    
    def test_create_empty_response_fallback(self):
        """Test creating empty response fallback for unknown model"""
        class UnknownModel:
            def __init__(self, **kwargs):
                self.candidate_files = kwargs.get('candidate_files', [])
        
        response = self.localizer._create_empty_response(UnknownModel)
        
        self.assertIsInstance(response, UnknownModel)
        self.assertEqual(response.candidate_files, [])
    
    def test_test_api_connectivity_success(self):
        """Test successful API connectivity test"""
        with patch.object(self.localizer, '_make_api_request', return_value="API connection successful"):
            result = self.localizer.test_api_connectivity()
            
            self.assertTrue(result)
    
    def test_test_api_connectivity_empty_response(self):
        """Test API connectivity test with empty response"""
        with patch.object(self.localizer, '_make_api_request', return_value=""):
            result = self.localizer.test_api_connectivity()
            
            self.assertFalse(result)
    
    def test_test_api_connectivity_error(self):
        """Test API connectivity test with error"""
        with patch.object(self.localizer, '_make_api_request', side_effect=Exception("Connection error")):
            result = self.localizer.test_api_connectivity()
            
            self.assertFalse(result)
    
    def test_check_model_availability_success(self):
        """Test successful model availability check"""
        with patch.object(self.localizer, '_make_api_request', return_value="Test response"):
            result = self.localizer.check_model_availability("qwen-coder-32b")
            
            self.assertTrue(result)
    
    def test_check_model_availability_not_available(self):
        """Test model availability check for unavailable model"""
        with patch.object(self.localizer, '_make_api_request', side_effect=ValueError("not currently available")):
            result = self.localizer.check_model_availability("unavailable-model")
            
            self.assertFalse(result)
    
    def test_check_model_availability_empty_response(self):
        """Test model availability check with empty response"""
        with patch.object(self.localizer, '_make_api_request', return_value=""):
            result = self.localizer.check_model_availability("qwen-coder-32b")
            
            self.assertFalse(result)
    
    def test_test_model_compatibility_success(self):
        """Test model compatibility testing with successful models"""
        test_models = ["qwen-coder-32b", "gpt-oss-20b"]
        
        with patch.object(self.localizer, '_make_api_request', return_value="This function calculates Fibonacci numbers."):
            results = self.localizer.test_model_compatibility(test_models)
            
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 2)
            
            for model_name in test_models:
                self.assertIn(model_name, results)
                self.assertEqual(results[model_name]["status"], "success")
                self.assertIn("model_id", results[model_name])
                self.assertIn("response_length", results[model_name])
    
    def test_test_model_compatibility_mixed_results(self):
        """Test model compatibility testing with mixed success/failure"""
        test_models = ["qwen-coder-32b", "invalid-model"]
        
        # Track which model is being tested
        call_count = 0
        
        def mock_api_request(prompt):
            nonlocal call_count
            call_count += 1
            # First call is for qwen-coder-32b (valid), second is for invalid-model
            if call_count == 1:
                return "This function calculates Fibonacci numbers."
            else:
                raise ValueError("Model not available")
        
        with patch.object(self.localizer, '_make_api_request', side_effect=mock_api_request):
            results = self.localizer.test_model_compatibility(test_models)
            
            self.assertEqual(results["qwen-coder-32b"]["status"], "success")
            self.assertEqual(results["invalid-model"]["status"], "error")
    
    def test_cleanup_success(self):
        """Test successful cleanup"""
        # Set some tracking attributes
        self.localizer._api_call_count = 5
        self.localizer._total_tokens_used = 1000
        self.localizer._successful_requests = 4
        self.localizer._failed_requests = 1
        
        # Test cleanup
        self.localizer.cleanup()
        
        # Verify cleanup actions
        self.assertIsNone(self.localizer.client)
        self.assertEqual(len(self.localizer.model_mapping), 0)
        self.assertIsNone(self.localizer.prompt_generator)
        self.assertIsNone(self.localizer.api_key)
    
    def test_cleanup_with_error(self):
        """Test cleanup handles errors gracefully"""
        # Mock an attribute that will cause an error during cleanup
        with patch.object(self.localizer, 'model_mapping', side_effect=Exception("Cleanup error")):
            # Should not raise exception
            self.localizer.cleanup()
    
    def test_destructor_calls_cleanup(self):
        """Test that destructor calls cleanup"""
        with patch.object(self.localizer, 'cleanup') as mock_cleanup:
            self.localizer.__del__()
            mock_cleanup.assert_called_once()
    
    def test_destructor_handles_cleanup_error(self):
        """Test that destructor handles cleanup errors silently"""
        with patch.object(self.localizer, 'cleanup', side_effect=Exception("Cleanup error")):
            # Should not raise exception
            self.localizer.__del__()


if __name__ == '__main__':
    unittest.main()