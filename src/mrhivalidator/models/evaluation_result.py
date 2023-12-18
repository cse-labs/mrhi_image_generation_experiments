from pydantic import BaseModel
from mrhivalidator.utils.criteria import CheckResult

class EvaluationResult(BaseModel):
    result: CheckResult
    score: float
    range: tuple
    explanation: str = None
    
