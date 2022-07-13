from importlib import import_module
from enum import Enum

class Scorer(Enum):
    BERTScore = 'scorer.scorer_bertscore.ScorerBERTScore'
    BLEU = 'scorer.scorer_bleu.ScorerBleu'
    CHRF = 'scorer.scorer_chrf.ScorerCHRF'
    COMET = 'scorer.scorer_comet.ScorerCOMET'
    ROUGE = 'scorer.scorer_rouge.ScorerRouge'
    TER = 'scorer.scorer_ter.ScorerTER'

def create_scorer( scorer: Scorer, source_language, target_language, verbose = False, **kwargs ):
    try:
        class_str = scorer.value
        module_path, class_name = class_str.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)( source_language = source_language, target_language = target_language, verbose = verbose, **kwargs )
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)
