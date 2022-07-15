import rouge

from scorer.scorer_base import ScorerBase
from utils.utils import Timer

# Please refer to rouge documentation to understand all the settings
# https://github.com/pltrdy/rouge
class ScorerRouge(ScorerBase):
    def __init__( self, source_language, target_language, verbose = False,
        metric = "rouge-l", return_lengths = False,
        raw_results = False, exclusive = True ) -> None:
        super(ScorerRouge, self).__init__( source_language, target_language, verbose = verbose )

        self.rouge_scorer = rouge.Rouge(
            metrics = [metric],
            return_lengths = return_lengths,
            raw_results = raw_results,
            exclusive = exclusive
        )

    def compute_score( self, source_list, prediction_list, reference_list ):
        output = self.rouge_scorer.get_scores( prediction_list, reference_list, avg=True, ignore_empty=True )
        output_dict = {
            "score": output[self.rouge_scorer.metrics[0]]['f'],
            "precision": output[self.rouge_scorer.metrics[0]]['p'],
            "recall": output[self.rouge_scorer.metrics[0]]['r'],

        }
        return output_dict

    @staticmethod
    def is_tokenization_required():
        """Does this particular scorer needs already pre-tokenized input or it will do tokenization itself"""
        return True

    @staticmethod
    def is_source_required():
        """Does this particular scorer needs original source text to do the scoring"""
        """Or just prediction and reference will be enough"""
        return False