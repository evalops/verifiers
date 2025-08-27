"""DSPy-based judges for Verifiers rubrics."""

import dspy

# Point DSPy to the same OpenAI-compatible endpoint as vLLM (or OpenAI proper)
# Example for local vLLM / SGLang:
# dspy.configure(lm=dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
#                           api_base="http://localhost:8000/v1",
#                           api_key="", model_type="chat"))

class RatingSig(dspy.Signature):
    """Rate the completion's correctness on a scale of 0.0 to 1.0. Respond with valid JSON containing 'score' (float) and 'rationale' (string)."""
    prompt      = dspy.InputField()
    completion  = dspy.InputField()
    answer      = dspy.InputField()
    score       = dspy.OutputField(desc="float in [0,1] representing correctness")
    rationale   = dspy.OutputField(desc="brief explanation of the score")

class DSPyJudge(dspy.Module):
    def __init__(self, use_json_adapter: bool = True):
        super().__init__()
        # JSONAdapter can have issues with some local models, make it optional
        if use_json_adapter:
            try:
                self.rate = dspy.ChainOfThought(RatingSig, adapter=dspy.JSONAdapter())
            except Exception:
                print("Warning: JSONAdapter failed, falling back to plain ChainOfThought")
                self.rate = dspy.ChainOfThought(RatingSig)
        else:
            self.rate = dspy.ChainOfThought(RatingSig)

    async def forward_async(self, prompt, completion, answer):
        """Async version for use in Verifiers rollouts."""
        import asyncio
        # Run the DSPy call in a thread to avoid blocking
        def _sync_call():
            out = self.rate(prompt=prompt, completion=completion, answer=answer)
            try:
                s = float(out.score)
            except Exception:
                s = 0.0
            # clamp
            s = max(0.0, min(1.0, s))
            return {"score": s, "rationale": out.rationale}
        
        return await asyncio.to_thread(_sync_call)

    def forward(self, prompt, completion, answer):
        """Sync version for standalone use."""
        out = self.rate(prompt=prompt, completion=completion, answer=answer)
        try:
            s = float(out.score)
        except (ValueError, AttributeError):
            # Fallback: try to extract score from rationale if JSON parsing failed
            try:
                rationale = str(getattr(out, 'rationale', out))
                import re
                score_match = re.search(r'score.*?(\d*\.?\d+)', rationale, re.IGNORECASE)
                if score_match:
                    s = float(score_match.group(1))
                    s = max(0.0, min(1.0, s))  # clamp to [0,1]
                else:
                    s = 0.5  # neutral score if can't parse
            except Exception:
                s = 0.0
        # clamp
        s = max(0.0, min(1.0, s))
        rationale = getattr(out, 'rationale', str(out))
        return {"score": s, "rationale": rationale}

# Verifiers rubric function (callable) - matches Rubric function signature
async def dspy_judge_reward_async(parser, completion, answer, state=None):
    """Async DSPy judge function for Verifiers rubrics."""
    # Create judge instance per call to avoid thread safety issues
    # Use plain ChainOfThought for better Ollama compatibility
    judge = DSPyJudge(use_json_adapter=False)
    
    # Extract prompt from state if available
    prompt = state.get("prompt", "") if state else ""
    
    res = await judge.forward_async(prompt=prompt, completion=completion, answer=answer)
    return res["score"]

def create_dspy_judge_reward_func():
    """Create a DSPy judge reward function with proper signature."""
    def reward_func(parser, completion, answer, state=None):
        """Sync wrapper for async judge - for testing only."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(dspy_judge_reward_async(parser, completion, answer, state))
        except RuntimeError:
            # No event loop running
            return asyncio.run(dspy_judge_reward_async(parser, completion, answer, state))
    
    return reward_func
