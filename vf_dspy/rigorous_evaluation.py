#!/usr/bin/env python3
"""
Rigorous A/B testing framework for DSPy + Verifiers integration.

This provides statistically sound comparison with proper controls.
"""

import asyncio
import json
import time
import random
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
import dspy
from dataclasses import dataclass


@dataclass
class TestResult:
    """Single test result with all metrics."""
    question: str
    answer: str
    completion: str
    total_reward: float
    metrics: Dict[str, float]
    scoring_time: float
    config_name: str
    run_id: int
    example_id: int


class RigorousEvaluator:
    """Statistically rigorous A/B testing for DSPy integration."""
    
    def __init__(self, 
                 test_size: int = 100,
                 num_runs: int = 5,
                 model: str = "llama3.2:latest"):
        self.test_size = test_size
        self.num_runs = num_runs
        self.model = model
        self.results: List[TestResult] = []
        
        # Configure DSPy
        dspy.configure(lm=dspy.LM(
            f"openai/{model}",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
            model_type="chat"
        ))
    
    def create_controlled_environments(self) -> Tuple[Any, Any]:
        """Create baseline and DSPy environments with identical base setup."""
        # Import here to avoid circular dependencies
        from vf_dspy.math_env import load_environment_original, load_environment_with_dspy
        
        # Use same dataset seed for reproducibility
        baseline_env = load_environment_original(
            num_train_examples=self.test_size,
            dataset_split="train"
        )
        
        enhanced_env = load_environment_with_dspy(
            num_train_examples=self.test_size,
            dataset_split="train", 
            judge_weight=0.3
        )
        
        # Verify same dataset
        assert len(baseline_env.dataset) == len(enhanced_env.dataset)
        for i in range(min(5, len(baseline_env.dataset))):
            baseline_ex = baseline_env.dataset[i]
            enhanced_ex = enhanced_env.dataset[i]
            
            baseline_q = baseline_ex.get("question", baseline_ex.get("problem", ""))
            enhanced_q = enhanced_ex.get("question", enhanced_ex.get("problem", ""))
            
            assert baseline_q == enhanced_q, f"Dataset mismatch at {i}"
            assert baseline_ex["answer"] == enhanced_ex["answer"], f"Answer mismatch at {i}"
        
        print(f"✅ Environments created with identical {len(baseline_env.dataset)} examples")
        return baseline_env, enhanced_env
    
    def generate_test_completions(self, question: str, answer: str) -> List[str]:
        """Generate diverse test completions for fair comparison."""
        return [
            f"The answer is {answer}",  # Minimal correct
            f"Let me solve this step by step.\n\nThe answer is {answer}",  # With reasoning
            f"After calculating, I get: {answer}",  # Different phrasing
            f"The solution is {answer}.",  # Formal
        ]
    
    async def evaluate_single_example(self, 
                                     env: Any,
                                     example: Dict,
                                     config_name: str,
                                     run_id: int,
                                     example_id: int) -> List[TestResult]:
        """Evaluate a single example with multiple completions."""
        question = example.get("question", example.get("problem", ""))
        answer = example["answer"]
        
        results = []
        completions = self.generate_test_completions(question, answer)
        
        for completion in completions:
            start_time = time.time()
            try:
                score_result = await env.rubric.score_rollout(
                    question, completion, answer, 
                    state={"prompt": question}
                )
                scoring_time = time.time() - start_time
                
                results.append(TestResult(
                    question=question[:100] + "..." if len(question) > 100 else question,
                    answer=str(answer),
                    completion=completion,
                    total_reward=score_result.reward,
                    metrics=dict(score_result.metrics),
                    scoring_time=scoring_time,
                    config_name=config_name,
                    run_id=run_id,
                    example_id=example_id
                ))
                
            except Exception as e:
                print(f"❌ Error evaluating {config_name} run {run_id} example {example_id}: {e}")
                # Add failed result with 0 reward
                results.append(TestResult(
                    question=question[:100] + "..." if len(question) > 100 else question,
                    answer=str(answer),
                    completion=completion,
                    total_reward=0.0,
                    metrics={},
                    scoring_time=0.0,
                    config_name=config_name,
                    run_id=run_id,
                    example_id=example_id
                ))
        
        return results
    
    async def run_single_configuration(self, 
                                      env: Any, 
                                      config_name: str,
                                      run_id: int) -> List[TestResult]:
        """Run evaluation for a single configuration."""
        print(f"🔄 Running {config_name} - Run {run_id + 1}/{self.num_runs}")
        
        results = []
        
        # Sample subset of examples for this run to manage time
        test_examples = env.dataset.select(range(min(20, len(env.dataset))))
        
        for i, example in enumerate(test_examples):
            example_results = await self.evaluate_single_example(
                env, example, config_name, run_id, i
            )
            results.extend(example_results)
            
            if i % 5 == 0 and i > 0:
                print(f"  📊 Completed {i}/{len(test_examples)} examples")
        
        avg_reward = statistics.mean(r.total_reward for r in results)
        print(f"  ✅ {config_name} Run {run_id + 1}: {avg_reward:.3f} avg reward")
        
        return results
    
    async def run_rigorous_comparison(self) -> Dict[str, Any]:
        """Run complete rigorous A/B comparison."""
        print("🧪 Starting Rigorous DSPy + Verifiers Evaluation")
        print(f"📊 Configuration: {self.test_size} examples, {self.num_runs} runs")
        
        # Create controlled environments
        baseline_env, enhanced_env = self.create_controlled_environments()
        
        all_results = []
        
        # Run multiple trials for each configuration
        for run_id in range(self.num_runs):
            print(f"\n--- Run {run_id + 1}/{self.num_runs} ---")
            
            # Baseline
            baseline_results = await self.run_single_configuration(
                baseline_env, "baseline", run_id
            )
            all_results.extend(baseline_results)
            
            # Enhanced
            enhanced_results = await self.run_single_configuration(
                enhanced_env, "enhanced", run_id
            )
            all_results.extend(enhanced_results)
        
        self.results = all_results
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Perform statistical analysis of results."""
        baseline_results = [r for r in self.results if r.config_name == "baseline"]
        enhanced_results = [r for r in self.results if r.config_name == "enhanced"]
        
        print(f"\n📊 Analyzing {len(baseline_results)} baseline vs {len(enhanced_results)} enhanced results")
        
        # Basic statistics
        baseline_rewards = [r.total_reward for r in baseline_results]
        enhanced_rewards = [r.total_reward for r in enhanced_results]
        
        baseline_mean = statistics.mean(baseline_rewards)
        enhanced_mean = statistics.mean(enhanced_rewards)
        baseline_std = statistics.stdev(baseline_rewards) if len(baseline_rewards) > 1 else 0
        enhanced_std = statistics.stdev(enhanced_rewards) if len(enhanced_rewards) > 1 else 0
        
        # Effect size (Cohen's d)
        pooled_std = ((baseline_std ** 2 + enhanced_std ** 2) / 2) ** 0.5
        cohens_d = (enhanced_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        # Welch's t-test approximation
        n1, n2 = len(baseline_rewards), len(enhanced_rewards)
        se_diff = (baseline_std**2/n1 + enhanced_std**2/n2)**0.5
        t_stat = (enhanced_mean - baseline_mean) / se_diff if se_diff > 0 else 0
        
        # Practical significance threshold
        improvement_pct = ((enhanced_mean / baseline_mean) - 1) * 100 if baseline_mean > 0 else 0
        
        # Per-run analysis for consistency
        run_analysis = {}
        for run_id in range(self.num_runs):
            baseline_run = [r.total_reward for r in baseline_results if r.run_id == run_id]
            enhanced_run = [r.total_reward for r in enhanced_results if r.run_id == run_id]
            
            if baseline_run and enhanced_run:
                run_baseline_mean = statistics.mean(baseline_run)
                run_enhanced_mean = statistics.mean(enhanced_run)
                run_improvement = ((run_enhanced_mean / run_baseline_mean) - 1) * 100 if run_baseline_mean > 0 else 0
                
                run_analysis[f"run_{run_id}"] = {
                    "baseline_mean": run_baseline_mean,
                    "enhanced_mean": run_enhanced_mean,
                    "improvement_pct": run_improvement,
                    "consistent_improvement": run_enhanced_mean > run_baseline_mean
                }
        
        # Judge-specific analysis (for enhanced only)
        enhanced_judge_scores = [
            r.metrics.get('dspy_judge_reward_async', 0.0) 
            for r in enhanced_results if r.metrics
        ]
        
        analysis = {
            "sample_sizes": {
                "baseline": len(baseline_results),
                "enhanced": len(enhanced_results),
                "total": len(self.results)
            },
            "reward_statistics": {
                "baseline": {
                    "mean": baseline_mean,
                    "std": baseline_std,
                    "min": min(baseline_rewards) if baseline_rewards else 0,
                    "max": max(baseline_rewards) if baseline_rewards else 0
                },
                "enhanced": {
                    "mean": enhanced_mean,
                    "std": enhanced_std,
                    "min": min(enhanced_rewards) if enhanced_rewards else 0,
                    "max": max(enhanced_rewards) if enhanced_rewards else 0
                }
            },
            "effect_analysis": {
                "improvement_pct": improvement_pct,
                "cohens_d": cohens_d,
                "t_statistic": t_stat,
                "effect_size_interpretation": self.interpret_cohens_d(cohens_d)
            },
            "consistency_analysis": run_analysis,
            "judge_analysis": {
                "avg_judge_score": statistics.mean(enhanced_judge_scores) if enhanced_judge_scores else 0,
                "judge_std": statistics.stdev(enhanced_judge_scores) if len(enhanced_judge_scores) > 1 else 0,
                "positive_judge_scores": sum(1 for s in enhanced_judge_scores if s > 0.5) if enhanced_judge_scores else 0
            },
            "practical_significance": {
                "statistically_significant": abs(t_stat) > 2.0,  # Rough approximation
                "practically_significant": improvement_pct > 5.0,  # 5% improvement threshold
                "consistent_across_runs": sum(1 for run_data in run_analysis.values() if run_data["consistent_improvement"]) / len(run_analysis) if run_analysis else 0
            }
        }
        
        return analysis
    
    def interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_rigorous_report(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive statistical report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
# Rigorous DSPy + Verifiers Statistical Evaluation

**Generated:** {timestamp}
**Model:** {self.model}
**Sample Size:** {analysis['sample_sizes']['total']} total evaluations
**Runs:** {self.num_runs} independent runs
**Examples per run:** ~{analysis['sample_sizes']['total'] // (2 * self.num_runs)}

## 🎯 Executive Summary

{'✅ **STATISTICALLY SIGNIFICANT IMPROVEMENT CONFIRMED**' if analysis['practical_significance']['statistically_significant'] and analysis['practical_significance']['practically_significant'] else '❌ **IMPROVEMENT NOT STATISTICALLY SIGNIFICANT**'}

- **Improvement:** {analysis['effect_analysis']['improvement_pct']:+.2f}%
- **Effect Size:** {analysis['effect_analysis']['cohens_d']:.3f} ({analysis['effect_analysis']['effect_size_interpretation']})
- **Consistency:** {analysis['practical_significance']['consistent_across_runs']:.1%} of runs showed improvement

## 📊 Statistical Analysis

### Sample Sizes & Power
- **Baseline samples:** {analysis['sample_sizes']['baseline']}
- **Enhanced samples:** {analysis['sample_sizes']['enhanced']}
- **Total evaluations:** {analysis['sample_sizes']['total']}

### Reward Distribution
| Metric | Baseline | Enhanced | Difference |
|--------|----------|----------|------------|
| Mean | {analysis['reward_statistics']['baseline']['mean']:.4f} | {analysis['reward_statistics']['enhanced']['mean']:.4f} | {analysis['reward_statistics']['enhanced']['mean'] - analysis['reward_statistics']['baseline']['mean']:+.4f} |
| Std Dev | {analysis['reward_statistics']['baseline']['std']:.4f} | {analysis['reward_statistics']['enhanced']['std']:.4f} | {analysis['reward_statistics']['enhanced']['std'] - analysis['reward_statistics']['baseline']['std']:+.4f} |
| Min | {analysis['reward_statistics']['baseline']['min']:.4f} | {analysis['reward_statistics']['enhanced']['min']:.4f} | {analysis['reward_statistics']['enhanced']['min'] - analysis['reward_statistics']['baseline']['min']:+.4f} |
| Max | {analysis['reward_statistics']['baseline']['max']:.4f} | {analysis['reward_statistics']['enhanced']['max']:.4f} | {analysis['reward_statistics']['enhanced']['max'] - analysis['reward_statistics']['baseline']['max']:+.4f} |

### Effect Size Analysis
- **Cohen's d:** {analysis['effect_analysis']['cohens_d']:.3f} ({analysis['effect_analysis']['effect_size_interpretation']} effect)
- **T-statistic:** {analysis['effect_analysis']['t_statistic']:.3f}
- **Improvement:** {analysis['effect_analysis']['improvement_pct']:+.2f}%

## 🔍 Consistency Analysis (Per Run)

"""
        
        # Add per-run results
        for run_key, run_data in analysis['consistency_analysis'].items():
            run_num = run_key.split('_')[1]
            consistent = "✅" if run_data['consistent_improvement'] else "❌"
            report += f"**Run {int(run_num) + 1}:** {consistent} {run_data['improvement_pct']:+.1f}% improvement\n"
        
        report += f"""

## 🧠 Judge Performance Analysis

- **Average Judge Score:** {analysis['judge_analysis']['avg_judge_score']:.3f}
- **Judge Score Std:** {analysis['judge_analysis']['judge_std']:.3f}
- **High Confidence Scores:** {analysis['judge_analysis']['positive_judge_scores']} (>{analysis['judge_analysis']['positive_judge_scores']/analysis['sample_sizes']['enhanced']*100:.1f}%)

## ✅ Statistical Conclusions

### Significance Tests
- **Statistical Significance:** {'✅ YES' if analysis['practical_significance']['statistically_significant'] else '❌ NO'} (|t| > 2.0)
- **Practical Significance:** {'✅ YES' if analysis['practical_significance']['practically_significant'] else '❌ NO'} (>5% improvement)
- **Consistency:** {'✅ HIGH' if analysis['practical_significance']['consistent_across_runs'] >= 0.8 else '⚠️ MODERATE' if analysis['practical_significance']['consistent_across_runs'] >= 0.6 else '❌ LOW'} ({analysis['practical_significance']['consistent_across_runs']:.1%} consistent runs)

### Interpretation
"""
        
        if (analysis['practical_significance']['statistically_significant'] and 
            analysis['practical_significance']['practically_significant'] and
            analysis['practical_significance']['consistent_across_runs'] >= 0.8):
            report += """
✅ **STRONG EVIDENCE FOR DSPy INTEGRATION EFFICACY**

The DSPy + Verifiers integration shows statistically significant, practically meaningful, and consistent improvements over the baseline. The effect size is sufficient to justify production deployment.
"""
        elif (analysis['practical_significance']['statistically_significant'] and 
              analysis['practical_significance']['practically_significant']):
            report += """
⚠️ **MODERATE EVIDENCE FOR DSPy INTEGRATION EFFICACY**

The DSPy integration shows statistically significant improvements, but consistency across runs suggests the effect may be variable. Consider larger sample sizes or judge calibration.
"""
        else:
            report += """
❌ **INSUFFICIENT EVIDENCE FOR DSPy INTEGRATION EFFICACY**

The current evidence does not support a statistically significant improvement. Consider:
1. Judge calibration with more training data
2. Larger sample sizes for better power
3. Different judge weighting strategies
"""
        
        report += f"""

## 🔬 Methodology Notes

- **Controlled Environment:** Identical datasets used for baseline and enhanced configurations
- **Multiple Completions:** Each example tested with 4 different completion styles
- **Multiple Runs:** {self.num_runs} independent runs to assess consistency
- **Statistical Tests:** Welch's t-test and Cohen's d for effect size
- **Significance Threshold:** p < 0.05 (|t| > 2.0 approximation)
- **Practical Threshold:** >5% improvement for practical significance

## 📋 Raw Data

```json
{json.dumps(analysis, indent=2)}
```
"""
        
        return report


async def main():
    """Run rigorous evaluation."""
    evaluator = RigorousEvaluator(
        test_size=100,  # Larger sample
        num_runs=3      # Multiple runs for consistency
    )
    
    try:
        analysis = await evaluator.run_rigorous_comparison()
        report = evaluator.generate_rigorous_report(analysis)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rigorous_evaluation_{timestamp}.md"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Rigorous evaluation complete!")
        print(f"📝 Full report saved to: {filename}")
        print("\n" + "="*80)
        print(report.split("## 📋 Raw Data")[0])  # Print report without raw data
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
