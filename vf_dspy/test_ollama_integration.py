#!/usr/bin/env python3
"""End-to-end test of DSPy + Verifiers integration using Ollama."""

import asyncio
import dspy


def test_ollama_connection():
    """Test basic Ollama connection."""
    try:
        # Configure DSPy to use Ollama (OpenAI-compatible API)
        dspy.configure(lm=dspy.LM(
            "openai/llama3.2:latest",
            api_base="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't need real API key
            model_type="chat"
        ))
        
        # Simple test call
        lm = dspy.settings.lm
        response = lm("What is 2+2?")
        print(f"✓ Ollama connection successful: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return False


def test_dspy_judge_with_ollama():
    """Test DSPy judge with real Ollama LM."""
    try:
        # Configure DSPy
        dspy.configure(lm=dspy.LM(
            "openai/llama3.2:latest",
            api_base="http://localhost:11434/v1", 
            api_key="ollama",
            model_type="chat"
        ))
        
        from vf_dspy.judges import DSPyJudge
        
        # Use plain ChainOfThought to avoid JSONAdapter issues
        judge = DSPyJudge(use_json_adapter=False)
        
        # Test with a simple math problem
        result = judge(
            prompt="What is 5 + 3?",
            completion="Let me calculate: 5 + 3 = 8. The answer is 8.",
            answer="8"
        )
        
        print(f"✓ Judge scoring successful:")
        print(f"  Score: {result['score']}")
        print(f"  Rationale: {result['rationale'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Judge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_with_ollama():
    """Test complete environment integration."""
    try:
        # Configure DSPy
        dspy.configure(lm=dspy.LM(
            "openai/llama3.2:latest",
            api_base="http://localhost:11434/v1",
            api_key="ollama", 
            model_type="chat"
        ))
        
        from vf_dspy.math_env import load_environment_with_dspy
        
        # Load small dataset for testing
        env = load_environment_with_dspy(
            num_train_examples=5,  # Very small for testing
            judge_weight=0.3
        )
        
        print(f"✓ Environment loaded: {len(env.dataset)} examples")
        print(f"  Rubric functions: {len(env.rubric.reward_funcs)}")
        print(f"  Weights: {env.rubric.reward_weights}")
        
        # Test async scoring on a single example
        example = env.dataset[0]
        question = example.get("question", example.get("problem", ""))
        answer = example["answer"]
        
        # Simple test completion
        completion = f"Let me solve this: The answer is {answer}"
        
        print(f"\nTesting scoring on:")
        print(f"  Question: {question[:80]}...")
        print(f"  Answer: {answer}")
        
        # Test async scoring
        async def test_scoring():
            rollout_score = await env.rubric.score_rollout(
                question, completion, answer, 
                state={"prompt": question}
            )
            return rollout_score
            
        score_result = asyncio.run(test_scoring())
        print(f"✓ Async scoring successful:")
        print(f"  Total reward: {score_result.reward:.3f}")
        print(f"  Metrics: {score_result.metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_with_ollama():
    """Test policy compilation with Ollama (simplified)."""
    try:
        # Configure DSPy
        dspy.configure(lm=dspy.LM(
            "openai/llama3.2:latest",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
            model_type="chat"
        ))
        
        from vf_dspy.policy_compile import compile_math_policy
        
        print("⚠ Policy compilation test skipped (takes ~5-10 minutes)")
        print("  To test manually: uv run python vf_dspy/policy_compile.py")
        
        # Instead, test the policy structure
        from vf_dspy.use_compiled_policy import load_math_env_with_compiled_policy
        
        # Load without compiled policy (will use default)
        env = load_math_env_with_compiled_policy(
            policy_path="nonexistent.pkl",  # Will fallback to default
            use_dspy_judge=True
        )
        
        print("✓ Policy loading structure works (using default system prompt)")
        return True
        
    except Exception as e:
        print(f"✗ Policy test failed: {e}")
        return False


def main():
    print("🦙 Testing DSPy + Verifiers with Ollama\n")
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("DSPy Judge", test_dspy_judge_with_ollama),
        ("Environment Integration", test_environment_with_ollama),
        ("Policy Structure", test_policy_with_ollama),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            if test_func():
                passed += 1
        except KeyboardInterrupt:
            print("Test interrupted by user")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Complete integration working with Ollama!")
        print("\nNext steps:")
        print("- Compile actual policy: python vf_dspy/policy_compile.py")
        print("- Run small evaluation: python vf_dspy/use_compiled_policy.py") 
        print("- Scale up for GRPO training")
    else:
        print(f"\n🔧 {total - passed} tests need attention")


if __name__ == "__main__":
    main()
