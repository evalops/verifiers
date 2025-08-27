"""Enhanced math-python environment with DSPy integration."""

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset
from verifiers.utils.tools import python
from vf_dspy.hook_rubric import create_math_rubric_with_dspy_judge


def load_environment_with_dspy(
    dataset_name: str = "math",
    dataset_split: str = "train", 
    num_train_examples: int = -1,
    judge_weight: float = 0.5,
    system_prompt: str = None,
    **kwargs,
):
    """Load math-python environment with DSPy judge integration."""
    
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    
    if system_prompt is None:
        system_prompt = "Use python for all calculations (variables do not persist). Give your answer inside \\boxed{}."
    
    # Create rubric with DSPy judge
    rubric, parser = create_math_rubric_with_dspy_judge(judge_weight=judge_weight)
    
    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=[python],
        max_turns=3,
        **kwargs,
    )
    
    return vf_env


def load_environment_original(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    system_prompt: str = None,
    rubric: vf.Rubric = None,
    parser: vf.Parser = None,
    **kwargs,
):
    """Load math-python environment with optional external rubric/parser."""
    from math_verify import parse, verify
    from verifiers.utils.data_utils import extract_boxed_answer
    
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    
    if system_prompt is None:
        system_prompt = "Use python for all calculations (variables do not persist). Give your answer inside \\boxed{}."
    
    # Use provided rubric/parser or create default ones
    if rubric is None or parser is None:
        if parser is None:
            parser = vf.Parser(extract_fn=extract_boxed_answer)
        
        if rubric is None:
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
                if isinstance(completion, str):
                    return 1.0
                try:
                    num_assistant_messages = len(parser.get_assistant_messages(completion))
                    return float(num_assistant_messages)
                except (TypeError, AttributeError):
                    return 1.0

            def num_tool_calls(parser, completion, answer, state=None) -> float:
                if isinstance(completion, str):
                    return 0.0
                try:
                    num_tool_calls = len(parser.get_tool_messages(completion))
                    return float(num_tool_calls)
                except (TypeError, AttributeError):
                    return 0.0

            def num_errors(parser, completion, answer, state=None) -> float:
                if isinstance(completion, str):
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
                    return 0.0

            rubric = vf.Rubric(
                funcs=[correct_answer_reward_func, num_turns, num_tool_calls, num_errors],
                weights=[1.0, 0.0, 0.0, -0.1],
                parser=parser,
            )

    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=[python],
        max_turns=3,
        **kwargs,
    )

    return vf_env
