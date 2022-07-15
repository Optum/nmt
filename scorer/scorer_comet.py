from comet import download_model, load_from_checkpoint
import torch

from scorer.scorer_base import ScorerBase
from utils.utils import Timer

# Please refer to BERTScore documentation to understand all the settings
# https://github.com/Tiiiger/bert_score
class ScorerCOMET(ScorerBase):
    def __init__( self, source_language, target_language, verbose = False,
        model_type = 'wmt20-comet-da',
        gpus = None,
        batch_size = 8 ) -> None:
        super(ScorerCOMET, self).__init__( source_language, target_language, verbose = verbose )

        model_path = download_model(model_type)
        self.comet_scorer = load_from_checkpoint(model_path)

        self.batch_size = batch_size

        if gpus is None:
            # Define do we need gpu or not if the number not set
            self.gpus = 1 if torch.cuda.is_available() else 0
        else:
            self.gpus = gpus

    def compute_score( self, source_list, prediction_list, reference_list ):
        data = [
            {
                "src": src,
                "mt": mt,
                "ref": ref
            } for (src,mt,ref) in zip(source_list,prediction_list,reference_list)
        ]

        # Comet specifically enables determenistic mode for torch,
        # but not everyone supports that. So we are storing previos value and restore it
        is_torch_deterministic = torch.are_deterministic_algorithms_enabled()
        is_torch_deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        seg_scores, sys_score = self.comet_scorer.predict( data, batch_size=self.batch_size, gpus=self.gpus )
        
        torch.use_deterministic_algorithms( is_torch_deterministic, warn_only = is_torch_deterministic_warn_only )

        output_dict = {
            "score": sys_score
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
        return True