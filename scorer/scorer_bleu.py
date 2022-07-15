import sacrebleu

from scorer.scorer_base import ScorerBase
from utils.utils import Timer

# Please refer to sacrebleu documentation to understand all the settings
# https://github.com/mjpost/sacrebleu
class ScorerBleu(ScorerBase):
    def __init__( self, source_language, target_language, verbose = False,
        lowercase = False, force = True,
        smooth_method = "exp", smooth_value = None,
        max_ngram_order = 4, effective_order = False ) -> None:
        super(ScorerBleu, self).__init__( source_language, target_language, verbose = verbose )

        # Sacrebleu ideology is that you need to use their own embedded tokenization for reproducibility of results
        # But in our experience it works strange on CJK languages, so we are prefer to use our own tokenization
        self.blue_scorer = sacrebleu.metrics.bleu.BLEU(
            lowercase = lowercase,
            force = force,
            tokenize = 'none',
            smooth_method = smooth_method,
            smooth_value = smooth_value,
            max_ngram_order = max_ngram_order,
            effective_order = effective_order,
            trg_lang = self.target_language.language
        )

    def compute_score( self, source_list, prediction_list, reference_list ):
        # Bleu supports multiple references per sentence. But for now we support only 1 reference
        output = self.blue_scorer.corpus_score( prediction_list, [reference_list] )
        output_dict = {
            "score": output.score,
            "counts": output.counts,
            "totals": output.totals,
            "precisions": output.precisions,
            "bp": output.bp,
            "sys_len": output.sys_len,
            "ref_len": output.ref_len,
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