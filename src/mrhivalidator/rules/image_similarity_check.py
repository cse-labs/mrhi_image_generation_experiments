from image_similarity_measures.evaluate import evaluation
from mrhivalidator.models.evaluation_result import EvaluationResult
from mrhivalidator.utils.base_check import BaseCheck
from mrhivalidator.utils.criteria import CheckResult

RULE_NAME = "ImageSimilarity"


class ImageSimilarity(BaseCheck):

    def evaluate(self, **kwargs) -> EvaluationResult:
        score = evaluation(org_img_path=kwargs['original_image'],
                           pred_img_path=kwargs['mrhi_image'],
                           metrics=["ssim"])

        score['ssim'] = score['ssim'] * 100

        result = CheckResult.PASS if score['ssim'] > 80 else CheckResult.FAIL
        print(f"Image Similarity: {score['ssim']:.2f}%")

        return EvaluationResult(result=result, score=score['ssim'], range=self.RANGE)
