# DSPy + Verifiers Integration

This package integrates DSPy with Verifiers for improved RL training on math-python environments.

## Quick Start

### 1. Install Dependencies

```bash
uv add six math-verify dspy-ai
# Already includes: pandas, numpy, openai, datasets, pydantic
```

### 2. Configure DSPy LM

Point DSPy to your OpenAI-compatible endpoint (same as vLLM):

```python
import dspy

# For OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))

# For local vLLM endpoint
dspy.configure(lm=dspy.LM(
    "openai/meta-llama/Meta-Llama-3-8B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="",
    model_type="chat"
))
```

### 3. Test Basic Integration

```bash
# Test basic integration (no LM needed)
uv run python test_integration.py

# Test DSPy judge integration with LM
uv run python -c "
import dspy
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', api_key='your-key'))
from vf_dspy.judges import DSPyJudge
judge = DSPyJudge()
result = judge('What is 2+2?', 'The answer is 4', '4')
print(result)
"
```

### 4. Pre-compile Policy (Optional)

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key="your-key"))

from vf_dspy.policy_compile import compile_math_policy
compiled = compile_math_policy()
```

### 5. Use Compiled Policy

```python
from vf_dspy.use_compiled_policy import load_math_env_with_compiled_policy, quick_eval_compiled_policy

# Load environment with compiled system prompt
env = load_math_env_with_compiled_policy()

# Quick evaluation
quick_eval_compiled_policy()
```

## Components

### DSPy Judge (`judges.py`)
- `DSPyJudge`: Uses CoT + JSONAdapter for structured scoring
- `dspy_judge_reward`: Hook for Verifiers rubrics
- Produces scores in [0,1] with rationale

### Policy Compilation (`policy_compile.py`) 
- Optimizes system prompts against environment rubric
- Uses MIPROv2 for instruction + few-shot optimization
- Saves to `artifacts/dspy_policy.pkl`

### Rubric Integration (`hook_rubric.py`)
- Combines existing math verification with DSPy judge
- Tunable weight for judge influence
- Drop-in replacement for existing rubrics

### Judge Calibration (`calibrate_judge.py`)
- Framework for calibrating judge against ground truth
- Requires training data: (prompt, completion, answer, score)
- Optimizes judge to match human/verified rewards

## Integration with GRPO

```python
# Use compiled system prompt in your environment
from vf_dspy.use_compiled_policy import load_math_env_with_compiled_policy
env = load_math_env_with_compiled_policy()

# Use DSPy judge in rubric
from vf_dspy.hook_rubric import create_math_rubric_with_dspy_judge
rubric, parser = create_math_rubric_with_dspy_judge(judge_weight=0.5)

env.rubric = rubric
env.parser = parser

# Proceed with normal GRPO training
```

## Next Steps

1. **Tool Enhancement**: Add PythonInterpreter sandbox (Upgrade #4)
2. **JSON Formatting**: Implement structured output parsing (Upgrade #3)  
3. **Data Generation**: Add CompleteAndGrounded filtering (Upgrade #5)
4. **Multi-turn Support**: Extend to MultiTurnEnv with format critics

## File Structure

```
vf_dspy/
├── __init__.py
├── README.md
├── judges.py              # DSPy judge implementation
├── policy_compile.py      # Policy optimization
├── use_compiled_policy.py # Integration with Verifiers
├── hook_rubric.py        # Rubric integration
└── calibrate_judge.py    # Judge calibration
```
