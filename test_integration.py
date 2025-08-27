#!/usr/bin/env python3
"""Test script for DSPy + Verifiers integration."""

import os
import sys

def test_basic_imports():
    """Test that all modules can be imported."""
    try:
        import dspy
        print("✓ DSPy imported successfully")
    except ImportError as e:
        print(f"✗ DSPy import failed: {e}")
        return False
        
    try:
        import verifiers as vf
        print("✓ Verifiers imported successfully")
    except ImportError as e:
        print(f"✗ Verifiers import failed: {e}")
        return False
        
    try:
        from vf_dspy.judges import DSPyJudge, dspy_judge_reward_async
        from vf_dspy.hook_rubric import create_math_rubric_with_dspy_judge
        print("✓ vf_dspy modules imported successfully")
    except ImportError as e:
        print(f"✗ vf_dspy import failed: {e}")
        return False
        
    return True

def test_environment_loading():
    """Test loading math-python environment."""
    try:
        from environments.math_python.math_python import load_environment
        env = load_environment()
        print(f"✓ Math environment loaded: {len(env.dataset)} examples")
        
        # Test a single example
        example = env.dataset[0]
        print(f"✓ Sample question: {example.get('question', example.get('problem', 'N/A'))[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Environment loading failed: {e}")
        return False

def test_dspy_judge_basic():
    """Test DSPy judge without LM configuration."""
    try:
        from vf_dspy.judges import DSPyJudge
        judge = DSPyJudge()
        print("✓ DSPy judge created (LM configuration needed for actual use)")
        return True
    except Exception as e:
        print(f"✗ DSPy judge creation failed: {e}")
        return False

def test_rubric_integration():
    """Test rubric integration."""
    try:
        from vf_dspy.hook_rubric import create_math_rubric_with_dspy_judge
        rubric, parser = create_math_rubric_with_dspy_judge(judge_weight=0.5)
        print(f"✓ Math rubric with DSPy judge created: {len(rubric.reward_funcs)} functions")
        return True
    except Exception as e:
        print(f"✗ Rubric integration failed: {e}")
        return False

def main():
    print("🧪 Testing DSPy + Verifiers Integration\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Environment Loading", test_environment_loading), 
        ("DSPy Judge Basic", test_dspy_judge_basic),
        ("Rubric Integration", test_rubric_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if test_func():
            passed += 1
        
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Ready for DSPy + Verifiers integration.")
        print("\nNext steps:")
        print("1. Configure DSPy LM: dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key='...'))")
        print("2. Run quick eval: python vf_dspy/use_compiled_policy.py")
        print("3. Compile policy: python vf_dspy/policy_compile.py")
    else:
        print(f"\n❌ {total - passed} tests failed. Check dependencies and setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
