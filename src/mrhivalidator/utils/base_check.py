from abc import ABC, abstractmethod
from mrhivalidator.models.evaluation_result import EvaluationResult

class BaseCheck(ABC):

    RANGE = (0, 100)  # Default range, can be overridden by subclass

    @abstractmethod
    def evaluate(self, **kwargs) -> EvaluationResult:
        pass
