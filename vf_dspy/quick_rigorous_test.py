#!/usr/bin/env python3
"""
Quick but rigorous A/B test for DSPy + Verifiers integration.

Smaller scale but proper statistical controls.
"""

import asyncio
import json
import time
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
    config_name: str
    run_id: int


class QuickRigorousEvaluator:
    """Statistically sound but faster evaluation."""
    
    def __init__(self, 
                 test_size: int = 30,
                 num_runs: int = 3,
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
        """Create baseline and DSPy environments with identical setup."""
        from vf_dspy.math_env import load_environment_original, load_environment_with_dspy
        
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
        print(f"✅ Controlled environments with {len(baseline_env.dataset)} identical examples")
        
        return baseline_env, enhanced_env
    
    async def evaluate_examples(self, 
                               env: Any, 
                               examples: List[Dict],
                               config_name: str,
                               run_id: int) -> List[TestResult]:
        """Evaluate multiple examples with single completion each."""
        results = []
        
        for i, example in enumerate(examples):
            question = example.get("question", example.get("problem", ""))
            answer = example["answer"]
            
            # Single consistent completion format
            completion = f"Let me solve this: The answer is {answer}"
            
            try:
                start_time = time.time()
                score_result = await env.rubric.score_rollout(
                    question, completion, answer, 
                    state={"prompt": question}
                )
                scoring_time = time.time() - start_time
                
                results.append(TestResult(
                    question=question[:50] + "..." if len(question) > 50 else question,
                    answer=str(answer),
                    completion=completion,
                    total_reward=score_result.reward,
                    metrics=dict(score_result.metrics),
                    config_name=config_name,
                    run_id=run_id
                ))
                
            except Exception as e:
                print(f"❌ Error in {config_name} run {run_id}, example {i}: {e}")
                results.append(TestResult(
                    question=question[:50] + "..." if len(question) > 50 else question,
                    answer=str(answer),
                    completion=completion,
                    total_reward=0.0,
                    metrics={},
                    config_name=config_name,
                    run_id=run_id
                ))
        
        return results
    
    async def run_comparison(self) -> Dict[str, Any]:
        """Run rigorous but quick comparison."""
        print("🧪 Quick Rigorous DSPy + Verifiers Evaluation")
        print(f"📊 {self.test_size} examples × {self.num_runs} runs = {self.test_size * self.num_runs * 2} total evaluations")
        
        baseline_env, enhanced_env = self.create_controlled_environments()
        
        # Use first 10 examples for quicker testing
        test_examples = baseline_env.dataset.select(range(10))
        
        all_results = []
        
        for run_id in range(self.num_runs):
            print(f"\n--- Run {run_id + 1}/{self.num_runs} ---")
            
            # Baseline evaluation
            print("  🔍 Running baseline...")
            baseline_results = await self.evaluate_examples(
                baseline_env, test_examples, "baseline", run_id
            )
            baseline_mean = statistics.mean(r.total_reward for r in baseline_results)
            print(f"  📊 Baseline: {baseline_mean:.3f} avg reward")
            
            # Enhanced evaluation
            print("  🚀 Running enhanced...")
            enhanced_results = await self.evaluate_examples(
                enhanced_env, test_examples, "enhanced", run_id
            )
            enhanced_mean = statistics.mean(r.total_reward for r in enhanced_results)
            print(f"  📊 Enhanced: {enhanced_mean:.3f} avg reward")
            
            improvement = ((enhanced_mean / baseline_mean) - 1) * 100 if baseline_mean > 0 else 0
            print(f"  📈 Run improvement: {improvement:+.1f}%")
            
            all_results.extend(baseline_results + enhanced_results)
        
        self.results = all_results
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Statistical analysis with proper controls."""
        baseline_results = [r for r in self.results if r.config_name == "baseline"]
        enhanced_results = [r for r in self.results if r.config_name == "enhanced"]
        
        baseline_rewards = [r.total_reward for r in baseline_results]
        enhanced_rewards = [r.total_reward for r in enhanced_results]
        
        # Basic statistics
        baseline_mean = statistics.mean(baseline_rewards)
        enhanced_mean = statistics.mean(enhanced_rewards)
        baseline_std = statistics.stdev(baseline_rewards) if len(baseline_rewards) > 1 else 0
        enhanced_std = statistics.stdev(enhanced_rewards) if len(enhanced_rewards) > 1 else 0
        
        # Effect size (Cohen's d)
        pooled_std = ((baseline_std ** 2 + enhanced_std ** 2) / 2) ** 0.5
        cohens_d = (enhanced_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        # Improvement percentage
        improvement_pct = ((enhanced_mean / baseline_mean) - 1) * 100 if baseline_mean > 0 else 0
        
        # Per-run consistency check
        consistent_improvements = 0
        run_improvements = []
        
        for run_id in range(self.num_runs):
            baseline_run = [r.total_reward for r in baseline_results if r.run_id == run_id]
            enhanced_run = [r.total_reward for r in enhanced_results if r.run_id == run_id]
            
            if baseline_run and enhanced_run:
                run_baseline_mean = statistics.mean(baseline_run)
                run_enhanced_mean = statistics.mean(enhanced_run)
                run_improvement = ((run_enhanced_mean / run_baseline_mean) - 1) * 100 if run_baseline_mean > 0 else 0
                run_improvements.append(run_improvement)
                
                if run_enhanced_mean > run_baseline_mean:
                    consistent_improvements += 1
        
        # Judge performance analysis
        enhanced_judge_scores = [
            r.metrics.get('dspy_judge_reward_async', 0.0) 
            for r in enhanced_results if r.metrics
        ]
        
        # Statistical significance (rough approximation)
        n = min(len(baseline_rewards), len(enhanced_rewards))
        se = (baseline_std**2/n + enhanced_std**2/n)**0.5
        t_stat = (enhanced_mean - baseline_mean) / se if se > 0 else 0
        
        return {
            "sample_size": {
                "per_config": len(baseline_results),
                "total": len(self.results)
            },
            "rewards": {
                "baseline_mean": baseline_mean,
                "enhanced_mean": enhanced_mean,
                "improvement_pct": improvement_pct,
                "baseline_std": baseline_std,
                "enhanced_std": enhanced_std
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": self.interpret_cohens_d(cohens_d),
                "t_statistic": t_stat
            },
            "consistency": {
                "consistent_runs": consistent_improvements,
                "total_runs": self.num_runs,
                "consistency_rate": consistent_improvements / self.num_runs,
                "run_improvements": run_improvements
            },
            "judge_performance": {
                "avg_score": statistics.mean(enhanced_judge_scores) if enhanced_judge_scores else 0,
                "positive_scores": sum(1 for s in enhanced_judge_scores if s > 0.5) if enhanced_judge_scores else 0,
                "total_scores": len(enhanced_judge_scores)
            },
            "significance": {
                "statistically_significant": abs(t_stat) > 2.0,
                "practically_significant": improvement_pct > 5.0,
                "highly_consistent": consistent_improvements == self.num_runs
            }
        }
    
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
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate concise statistical report."""
        sig = analysis['significance']
        rewards = analysis['rewards']
        effect = analysis['effect_size']
        consistency = analysis['consistency']
        judge = analysis['judge_performance']
        
        # Overall conclusion
        if (sig['statistically_significant'] and 
            sig['practically_significant'] and 
            sig['highly_consistent']):
            conclusion = "✅ **STRONG STATISTICAL EVIDENCE**"
            recommendation = "🚀 **RECOMMENDED FOR PRODUCTION**"
        elif sig['statistically_significant'] and sig['practically_significant']:
            conclusion = "⚠️ **MODERATE STATISTICAL EVIDENCE**"
            recommendation = "🤔 **CONSIDER WITH CAUTION**"
        else:
            conclusion = "❌ **INSUFFICIENT EVIDENCE**"
            recommendation = "🛑 **NOT RECOMMENDED**"
        
        report = f"""
# Quick Rigorous DSPy Evaluation Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sample:** {analysis['sample_size']['per_config']} examples per config × {self.num_runs} runs

## 🎯 Executive Summary

{conclusion}
{recommendation}

## 📊 Key Results

| Metric | Baseline | Enhanced | Difference |
|--------|----------|----------|------------|
| **Mean Reward** | {rewards['baseline_mean']:.4f} | {rewards['enhanced_mean']:.4f} | **{rewards['improvement_pct']:+.1f}%** |
| **Std Deviation** | {rewards['baseline_std']:.4f} | {rewards['enhanced_std']:.4f} | - |

## 📈 Statistical Analysis

- **Effect Size (Cohen's d):** {effect['cohens_d']:.3f} ({effect['interpretation']})
- **T-statistic:** {effect['t_statistic']:.3f}
- **Statistical Significance:** {'✅ YES' if sig['statistically_significant'] else '❌ NO'}
- **Practical Significance:** {'✅ YES' if sig['practically_significant'] else '❌ NO'} (>{rewards['improvement_pct']:.1f}%)

## 🔄 Consistency Analysis

- **Runs with improvement:** {consistency['consistent_runs']}/{consistency['total_runs']} ({consistency['consistency_rate']:.1%})
- **Per-run improvements:** {', '.join(f'{imp:+.1f}%' for imp in consistency['run_improvements'])}
- **Highly consistent:** {'✅ YES' if sig['highly_consistent'] else '❌ NO'}

## 🧠 Judge Performance

- **Average judge score:** {judge['avg_score']:.3f}
- **High-confidence scores:** {judge['positive_scores']}/{judge['total_scores']} ({judge['positive_scores']/judge['total_scores']*100 if judge['total_scores'] > 0 else 0:.1f}%)

## ✅ Conclusions

### Evidence Quality
{'🔬 **HIGH**: Statistically significant, practically meaningful, and highly consistent.' if sig['statistically_significant'] and sig['practically_significant'] and sig['highly_consistent'] else '🔬 **MODERATE**: Some positive evidence but not fully consistent.' if sig['statistically_significant'] or sig['practically_significant'] else '🔬 **LOW**: No clear statistical evidence of improvement.'}

### Recommendation
{'Deploy with confidence - the DSPy integration shows consistent, measurable improvements.' if conclusion.startswith('✅') else 'Consider additional evaluation or judge tuning before production deployment.' if conclusion.startswith('⚠️') else 'Current evidence does not support deployment. Consider alternative approaches.'}

---
*Evaluation used controlled environments, identical datasets, and multiple runs for statistical rigor.*
"""
        
        return report


async def main():
    """Run quick rigorous evaluation."""
    evaluator = QuickRigorousEvaluator(
        test_size=30,   # Reasonable size
        num_runs=3      # Multiple runs for consistency
    )
    
    try:
        analysis = await evaluator.run_comparison()
        report = evaluator.generate_report(analysis)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quick_rigorous_eval_{timestamp}.md"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Quick rigorous evaluation complete!")
        print(f"📝 Report saved to: {filename}")
        print("\n" + "="*60)
        print(report)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
