import bert_score

from scorer.scorer_base import ScorerBase
from utils.utils import Timer

# Please refer to BERTScore documentation to understand all the settings
# https://github.com/Tiiiger/bert_score
class ScorerBERTScore(ScorerBase):
    def __init__( self, source_language, target_language, verbose = False,
        model_type = 'xlm-roberta-large',
        num_layers = None, all_layers = False,
        batch_size = 8, nthreads = 4,
        rescale_with_baseline = True,
        use_fast_tokenizer = True,
        device = None ) -> None:
        super(ScorerBERTScore, self).__init__( source_language, target_language, verbose = verbose )

        self.bert_scorer = bert_score.BERTScorer(
            lang = self.target_language.language,
            model_type = model_type,
            num_layers = num_layers,
            batch_size = batch_size,
            nthreads = nthreads,
            all_layers = all_layers,
            device = device,
            rescale_with_baseline = rescale_with_baseline,
            use_fast_tokenizer = use_fast_tokenizer
        )

    def compute_score( self, source_list, prediction_list, reference_list ):
        (P, R, F) = self.bert_scorer.score(
            cands=prediction_list,
            refs=reference_list,
            batch_size = self.bert_scorer.batch_size
        )
        output_dict = {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "score": F.mean().item()
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