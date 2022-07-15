import abc

from utils.utils import get_logger

class ScorerBase(abc.ABC):
    def __init__( self, source_language, target_language, verbose ) -> None:
        self.logger = get_logger( self.__class__.__name__, verbose )

        self.source_language = source_language
        self.target_language = target_language

    @abc.abstractmethod
    def compute_score( self, source_list, prediction_list, reference_list ):
        """Method to do actual scoring of the predictions"""

    @staticmethod
    @abc.abstractmethod
    def is_tokenization_required():
        """Does this particular scorer needs already pre-tokenized input or it will do tokenization itself"""
        return False

    @staticmethod
    @abc.abstractmethod
    def is_source_required():
        """Does this particular scorer needs original source text to do the scoring"""
        """Or just prediction and reference will be enough"""
        return False