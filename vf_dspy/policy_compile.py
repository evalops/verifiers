"""Compile a DSPy policy against math-python environment rubric."""

import dspy
import verifiers as vf
from dspy import Example

# 0) Align DSPy LM with your vLLM endpoint
# dspy.configure(lm=dspy.LM("openai/<your-model>", api_base="http://localhost:8000/v1",
#                           api_key="", model_type="chat"))  # same stack as vf-vllm

def compile_math_policy():
    # Load the math-python environment (original version without DSPy judge for compilation)
    from vf_dspy.math_env import load_environment_original
    env = load_environment_original()

    # 1) define a tiny "program" for the policy
    class SolveSig(dspy.Signature):
        """Solve mathematical problems step by step. Use available python tools for calculations. Give your final answer inside \\boxed{}."""
        question = dspy.InputField()
        answer   = dspy.OutputField(desc="step-by-step solution with final numeric answer inside \\boxed{}")

    class Policy(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought(SolveSig)  # CoT is a strong default

        def forward(self, question):
            return self.solve(question=question)

    # 2) build a metric that uses async scoring properly
    async def score_async(question, completion, answer):
        """Score a single example using the environment's rubric."""
        rollout_score = await env.rubric.score_rollout(question, completion, answer, state={"prompt": question})
        return rollout_score.reward

    def metric(prog: Policy, batch):
        """Metric function for DSPy optimization."""
        import asyncio
        
        async def _score_batch():
            tasks = []
            for ex in batch:
                pred = prog(question=ex.question).answer
                tasks.append(score_async(ex.question, pred, ex.gold))
            
            scores = await asyncio.gather(*tasks)
            return sum(scores) / max(1, len(scores))
        
        # Run the async scoring
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_score_batch())
        except RuntimeError:
            return asyncio.run(_score_batch())

    # 3) convert a small slice of env dataset to DSPy Examples
    ds = env.dataset.select(range(50))  # smaller for faster compilation
    train = []
    for x in ds:
        question = x.get("question", x.get("problem", ""))
        if question:  # only add if we have a valid question
            train.append(Example(question=question, gold=x["answer"]).with_inputs("question"))
    
    if not train:
        raise ValueError("No valid training examples found in dataset")

    # 4) optimize prompt+few-shots for the policy
    opt = dspy.MIPROv2()
    policy = Policy()
    
    import os
    os.makedirs("artifacts", exist_ok=True)
    
    compiled = opt.compile(policy, trainset=train, metric=metric)
    compiled.save("artifacts/dspy_policy.pkl", save_program=True)
    
    return compiled

if __name__ == "__main__":
    import os
    os.makedirs("artifacts", exist_ok=True)
    
    # You need to configure DSPy with your LM first
    print("Please configure DSPy with your language model first:")
    print('dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))')
    print("Then run: compiled = compile_math_policy()")
