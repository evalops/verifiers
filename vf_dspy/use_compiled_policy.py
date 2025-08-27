"""Use compiled DSPy policy in Verifiers environment."""

import dspy
import verifiers as vf
from environments.math_python.math_python import load_environment

def load_math_env_with_compiled_policy(policy_path: str = "artifacts/dspy_policy.pkl", use_dspy_judge: bool = True):
    """Load math environment with compiled DSPy system prompt."""
    
    system_prompt = None
    try:
        policy = dspy.load(policy_path)  # compiled DSPy program
        # Try different attribute paths for compiled instructions
        system_prompt = getattr(policy.solve, "instructions", None) or getattr(policy.solve.signature, "instructions", None)
        if system_prompt:
            print(f"Using compiled system prompt: {system_prompt[:100]}...")
        else:
            print("Compiled policy loaded but no instructions found, using default")
    except Exception as e:
        print(f"Could not load compiled policy: {e}")
        print("Using default system prompt")
    
    if system_prompt is None:
        system_prompt = "Use python for all calculations (variables do not persist). Give your answer inside \\boxed{}."

    # Choose environment loader based on whether to use DSPy judge
    if use_dspy_judge:
        from vf_dspy.math_env import load_environment_with_dspy
        env = load_environment_with_dspy(system_prompt=system_prompt)
    else:
        from vf_dspy.math_env import load_environment_original
        env = load_environment_original(system_prompt=system_prompt)
    
    return env

def quick_eval_compiled_policy():
    """Quick evaluation of compiled policy."""
    env = load_math_env_with_compiled_policy()
    
    # sanity check: small eval before GRPO
    from openai import OpenAI
    client = OpenAI()
    result = env.evaluate(client, "gpt-4o-mini", num_examples=10, rollouts_per_example=2)
    
    print("Evaluation results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result

if __name__ == "__main__":
    quick_eval_compiled_policy()
