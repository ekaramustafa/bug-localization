#!/usr/bin/env python3
"""
Test runner for OpenRouterLocalizer comprehensive unit tests

This script runs all test suites for the OpenRouterLocalizer implementation:
1. Unit tests for core functionality
2. Integration tests with sample bug instances  
3. Hierarchical file selection tests

Usage:
    python run_openrouter_tests.py [--verbose] [--suite SUITE_NAME]

Options:
    --verbose: Enable verbose test output
    --suite: Run specific test suite (unit, integration, hierarchical, all)
"""

import unittest
import sys
import os
import argparse
from io import StringIO

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import test modules
import test_openrouter_localizer
import test_openrouter_integration  
import test_openrouter_hierarchical


def run_test_suite(suite_name, verbose=False):
    """Run a specific test suite
    
    Args:
        suite_name: Name of the test suite to run ('unit', 'integration', 'hierarchical', 'all')
        verbose: Whether to enable verbose output
        
    Returns:
        TestResult object with test results
    """
    # Configure test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    # Create test suite based on requested suite
    if suite_name == 'unit':
        suite = unittest.TestLoader().loadTestsFromModule(test_openrouter_localizer)
        print("Running OpenRouterLocalizer Unit Tests...")
        
    elif suite_name == 'integration':
        suite = unittest.TestLoader().loadTestsFromModule(test_openrouter_integration)
        print("Running OpenRouterLocalizer Integration Tests...")
        
    elif suite_name == 'hierarchical':
        suite = unittest.TestLoader().loadTestsFromModule(test_openrouter_hierarchical)
        print("Running OpenRouterLocalizer Hierarchical Selection Tests...")
        
    elif suite_name == 'all':
        # Combine all test suites
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromModule(test_openrouter_localizer))
        suite.addTests(unittest.TestLoader().loadTestsFromModule(test_openrouter_integration))
        suite.addTests(unittest.TestLoader().loadTestsFromModule(test_openrouter_hierarchical))
        print("Running All OpenRouterLocalizer Tests...")
        
    else:
        raise ValueError(f"Unknown test suite: {suite_name}")
    
    # Run the tests
    print(f"{'='*60}")
    result = runner.run(suite)
    print(f"{'='*60}")
    
    return result


def print_test_summary(results):
    """Print a summary of test results
    
    Args:
        results: List of TestResult objects from each test suite
    """
    total_tests = sum(result.testsRun for result in results)
    total_failures = sum(len(result.failures) for result in results)
    total_errors = sum(len(result.errors) for result in results)
    total_skipped = sum(len(result.skipped) for result in results)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Successes: {total_tests - total_failures - total_errors - total_skipped}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    
    if total_failures > 0 or total_errors > 0:
        print(f"\nOVERALL RESULT: FAILED")
        return False
    else:
        print(f"\nOVERALL RESULT: PASSED")
        return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Run OpenRouterLocalizer tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose test output')
    parser.add_argument('--suite', '-s', choices=['unit', 'integration', 'hierarchical', 'all'],
                       default='all', help='Test suite to run')
    
    args = parser.parse_args()
    
    print("OpenRouterLocalizer Test Suite")
    print("="*60)
    print(f"Running test suite: {args.suite}")
    print(f"Verbose output: {args.verbose}")
    print()
    
    try:
        if args.suite == 'all':
            # Run all test suites individually for better organization
            results = []
            
            print("1. Running Unit Tests...")
            result = run_test_suite('unit', args.verbose)
            results.append(result)
            
            print("\n2. Running Integration Tests...")
            result = run_test_suite('integration', args.verbose)
            results.append(result)
            
            print("\n3. Running Hierarchical Selection Tests...")
            result = run_test_suite('hierarchical', args.verbose)
            results.append(result)
            
            # Print overall summary
            success = print_test_summary(results)
            
        else:
            # Run single test suite
            result = run_test_suite(args.suite, args.verbose)
            results = [result]
            success = print_test_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()