import sacremoses
import stanza
import langcodes

from utils.utils import Timer, get_logger
from scorer.create_scorer import Scorer, create_scorer

class DatasetScorer():
    def __init__(
        self,
        source_language,
        target_language,
        required_scorers = [Scorer.BERTScore,Scorer.BLEU,Scorer.CHRF,Scorer.COMET,Scorer.ROUGE,Scorer.TER],
        verbose = False,
        scorer_configs = {}
    ):
        self.logger = get_logger( self.__class__.__name__, verbose )

        self.source_language = langcodes.Language.get( source_language )
        self.target_language = langcodes.Language.get( target_language )

        self.is_tokenization_required = False
        self.is_source_required = False

        self.scorers = {}
        for scorer_class in required_scorers:
            with Timer( f'Creating scorer {scorer_class.name}', self.logger ):
                scorer_config = scorer_configs[scorer_class.name] if scorer_class.name in scorer_configs else {}
                scorer = create_scorer( scorer_class, self.source_language, self.target_language, verbose, **scorer_config )
                self.scorers[scorer_class.name] = scorer
                self.is_tokenization_required = self.is_tokenization_required or scorer.is_tokenization_required()
                self.is_source_required = self.is_source_required or scorer.is_source_required()

        self.src_list = None
        self.src_tokens = None
        self.ref_list = None
        self.ref_tokens = None

        with Timer( 'Normalization initialization', self.logger ):
            self.trg_normalizer = sacremoses.MosesPunctNormalizer(
                lang=self.target_language.language, pre_replace_unicode_punct=True, post_remove_control_chars=True )
            self.src_normalizer = sacremoses.MosesPunctNormalizer(
                lang=self.source_language.language, pre_replace_unicode_punct=True, post_remove_control_chars=True )

        if self.is_tokenization_required:
            with Timer( 'Target tokenizer initialization', self.logger ):
                self.trg_tokenizer = stanza.Pipeline(self.target_language.language,
                    processors='tokenize', tokenize_no_ssplit=True, verbose = verbose )

        if self.is_tokenization_required and self.is_source_required:
            with Timer( 'Source tokenizer initialization', self.logger ):
                self.src_tokenizer = stanza.Pipeline(self.source_language.language,
                    processors='tokenize', tokenize_no_ssplit=True, verbose = verbose )

    def set_source_and_reference( self, source_file, ref_file ):
        if self.is_source_required:
            with Timer( 'Preparing source data', self.logger ):
                with open( source_file, 'r', encoding='utf8' ) as src:
                    self.src_list = [self.src_normalizer.normalize( line.strip() ) for line in src]
                    # Some scorers break on completly empty lines
                    self.src_list = [line if line != '' else ' ' for line in self.src_list]
        else:
            self.src_list = None

        with Timer( 'Preparing reference data', self.logger ):
            with open( ref_file, 'r', encoding='utf8' ) as ref:
                self.ref_list = [self.trg_normalizer.normalize( line.strip() ) for line in ref]
                # Some scorers break on completly empty lines
                self.ref_list = [line if line != '' else ' ' for line in self.ref_list]

        if self.is_source_required and len(self.ref_list) != len(self.src_list):
            raise Exception(
                f"Source data {source_file} has different length from reference data {ref_file}."
            )

        if self.is_tokenization_required:
            with Timer( 'Performing reference tokenization', self.logger ):
                self.ref_tokens = [self._tokenize_text(text,self.trg_tokenizer) for text in self.ref_list]
        else:
            self.ref_tokens = None

        if self.is_tokenization_required and self.is_source_required:
            with Timer( 'Performing source tokenization', self.logger ):
                self.src_tokens = [self._tokenize_text(text,self.src_tokenizer) for text in self.src_list]
        else:
            self.src_tokens = None

    def score_translations( self, hypotesis_file ):
        if not self.ref_list:
            raise Exception(
                f"Cannot score {hypotesis_file} because reference file is not set"
            )

        with Timer( 'Preparing hypotesis data', self.logger ):
            with open( hypotesis_file, 'r', encoding='utf8' ) as hyp:
                hyp_list = [self.trg_normalizer.normalize( line.strip() ) for line in hyp]
                # Some scorers break on completly empty lines
                hyp_list = [line if line != '' else ' ' for line in hyp_list]

        if len(self.ref_list) != len(hyp_list):
            raise Exception(
                f"Hypothesis data {hypotesis_file} has different length from reference data."
            )

        if self.is_tokenization_required:
            with Timer( 'Performing hypotesis tokenization', self.logger ):
                hyp_tokens = [self._tokenize_text(text,self.trg_tokenizer) for text in hyp_list]

        result = {}
        for scorer_name,scorer in self.scorers.items():
            with Timer( f'Scoring data with {scorer_name}', self.logger ):
                if self.is_tokenization_required:
                    score = scorer.compute_score( self.src_tokens, hyp_tokens, self.ref_tokens )
                else:
                    score = scorer.compute_score( self.src_list, hyp_list, self.ref_list )
                result[scorer_name] = score['score']

        return result

    @staticmethod
    def _tokenize_text( text, tokenizer ):
        tokens = ' '.join( [token.text for sentence in tokenizer(text).sentences for token in sentence.tokens ] )
        # Rouge breaks on lines that consists entirely of full stops, so filter them there
        tokens = tokens if tokens.replace('.','') != '' else ''
        # Some scorers break on completly empty lines
        tokens = tokens if tokens != '' else ' '
        return tokens
