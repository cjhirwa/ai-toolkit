"""evalh — score any pipeline against a dataset and track runs over time."""

from .compare import Comparison, compare
from .dataset import Example, load_dataset, save_dataset
from .metrics import (
    BUILTIN,
    Contains,
    ExactMatch,
    FunctionMetric,
    JsonValid,
    LLMJudge,
    Metric,
    Output,
    Regex,
    Score,
    TokenF1,
    answer_relevance,
    correctness,
    faithfulness,
    judge,
    metric,
)
from .report import comparison_report, run_report
from .runner import ExampleResult, RunResult, run_eval
from .store import Store

__all__ = [
    "Example", "load_dataset", "save_dataset", "Output", "Score", "Metric", "BUILTIN",
    "ExactMatch", "Contains", "Regex", "TokenF1", "JsonValid", "FunctionMetric", "metric",
    "LLMJudge", "judge", "faithfulness", "answer_relevance", "correctness",
    "run_eval", "RunResult", "ExampleResult", "Store", "compare", "Comparison",
    "run_report", "comparison_report",
]
