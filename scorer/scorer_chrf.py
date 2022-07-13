import sacrebleu

from scorer.scorer_base import ScorerBase
from utils.utils import Timer

# Please refer to sacrebleu documentation to understand all the settings
# https://github.com/mjpost/sacrebleu
class ScorerCHRF(ScorerBase):
    def __init__( self, source_language, target_language, verbose = False,
        char_order = 6, word_order = 0,
        beta = 2, lowercase = False,
        whitespace = False, eps_smoothing = False ) -> None:
        super(ScorerCHRF, self).__init__( source_language, target_language, verbose = verbose )

        self.chrf_scorer = sacrebleu.metrics.chrf.CHRF(
            char_order = char_order,
            word_order = word_order,
            beta = beta,
            lowercase = lowercase,
            whitespace = whitespace,
            eps_smoothing = eps_smoothing
        )

    def compute_score( self, source_list, prediction_list, reference_list ):
        # CHRF supports multiple references per sentence. But for now we support only 1 reference
        output = self.chrf_scorer.corpus_score( prediction_list, [reference_list] )
        output_dict = {
            "score": output.score,
            "char_order": output.char_order,
            "word_order": output.word_order,
            "beta": output.beta,
        }
        return output_dict

    @staticmethod
    def is_tokenization_required():
        """Does this particular scorer needs already pre-tokenized input or it will do tokenization itself"""
        return False

    @staticmethod
    def is_source_required():
        """Does this particular scorer needs original source text to do the scoring"""
        """Or just prediction and reference will be enough"""
        return False

    @staticmethod
    def scorer_name():
        """Name of the scorer"""
        return 'CHRF'