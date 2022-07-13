import sacrebleu

from scorer.scorer_base import ScorerBase
from utils.utils import Timer

# Please refer to sacrebleu documentation to understand all the settings
# https://github.com/mjpost/sacrebleu
class ScorerTER(ScorerBase):
    def __init__( self, source_language, target_language, verbose = False,
        normalized = False, no_punct = False,
        asian_support = False, case_sensitive = False ) -> None:
        super(ScorerTER, self).__init__( source_language, target_language, verbose = verbose )

        # Sacrebleu ideology is that you need to use their own embedded tokenization for reproducibility of results
        # But in our experience it works strange on CJK languages, so we are prefer to use our own tokenization
        self.ter_scorer = sacrebleu.metrics.ter.TER(
            normalized = normalized,
            no_punct = no_punct,
            asian_support = asian_support,
            case_sensitive = case_sensitive
        )

    def compute_score( self, source_list, prediction_list, reference_list ):
        # TER supports multiple references per sentence. But for now we support only 1 reference
        output = self.ter_scorer.corpus_score( prediction_list, [reference_list] )
        output_dict = {
            "score": output.score,
            "num_edits": output.num_edits,
            "ref_length": output.ref_length
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

    @staticmethod
    def scorer_name():
        """Name of the scorer"""
        return 'TER'