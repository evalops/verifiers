"""Hook DSPy judge into Verifiers rubric."""

import verifiers as vf
from vf_dspy.judges import dspy_judge_reward_async

def create_math_rubric_with_dspy_judge(judge_weight: float = 0.5):
    """Create a rubric that includes both existing math verification and DSPy judge."""
    from math_verify import parse, verify
    from verifiers.utils.data_utils import extract_boxed_answer
    
    parser = vf.Parser(extract_fn=extract_boxed_answer)
    
    def correct_answer_reward_func(parser, completion, answer, state=None) -> float:
        completion_answer = parser.parse_answer(completion)
        parsed_completion_answer = parse(completion_answer, parsing_timeout=0)
        parsed_ground_truth_answer = parse(answer, parsing_timeout=0)
        if verify(
            parsed_completion_answer, parsed_ground_truth_answer, timeout_seconds=0
        ):
            return 1.0
        else:
            return 0.0
    
    def num_turns(parser, completion, answer, state=None) -> float:
        # Handle both string and list[dict] completion formats
        if isinstance(completion, str):
            return 1.0  # Single turn for string completions
        try:
            num_assistant_messages = len(parser.get_assistant_messages(completion))
            return float(num_assistant_messages)
        except (TypeError, AttributeError):
            return 1.0  # Fallback

    def num_tool_calls(parser, completion, answer, state=None) -> float:
        # Handle both string and list[dict] completion formats
        if isinstance(completion, str):
            return 0.0  # No tool calls in string completions
        try:
            num_tool_calls = len(parser.get_tool_messages(completion))
            return float(num_tool_calls)
        except (TypeError, AttributeError):
            return 0.0  # Fallback

    def num_errors(parser, completion, answer, state=None) -> float:
        # Handle both string and list[dict] completion formats
        if isinstance(completion, str):
            # Check for error keywords in string completion
            return 1.0 if "error" in completion.lower() else 0.0
        try:
            num_errors = sum(
                [
                    1.0
                    for msg in parser.get_tool_messages(completion)
                    if "error" in msg["content"].lower()
                ]
            )
            return float(num_errors)
        except (TypeError, AttributeError):
            return 0.0  # Fallback
    
    rubric = vf.Rubric(
        funcs=[
            correct_answer_reward_func,      # exact math verification
            dspy_judge_reward_async,         # DSPy judge (async)
            num_turns,
            num_tool_calls, 
            num_errors
        ],
        weights=[1.0, judge_weight, 0.0, 0.0, -0.1],  # tune judge weight
        parser=parser,
    )
    
    return rubric, parser
