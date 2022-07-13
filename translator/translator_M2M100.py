import torch.cuda
import langcodes
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language

class TranslatorM2M100( TranslatorBase ):
    def __init__( self, large_model = False, device = None,
        max_batch_lines = 4, max_batch_chars = 2000, max_file_lines = 60,
        verbose = False ):

        max_batch_lines = max( 1, max_batch_lines / 2 ) if large_model else max_batch_lines

        super(TranslatorM2M100, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        self.logger.info(f'Cuda is avaliable: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            self.logger.info(f'Number of avaliable GPUs: {torch.cuda.device_count()}')

        model_name = 'facebook/m2m100_1.2B' if large_model else 'facebook/m2m100_418M'

        with Timer( f'M2M100 model initialization', self.logger ):
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        self.model.eval()
        if device:
            self.model.to( device )
        elif torch.cuda.is_available():
            self.model.to('cuda')

        self.supported_languages = { langcodes.Language.get( lang ): lang for lang in self.tokenizer.lang_code_to_token.keys() }

        self.logger.info(f'M2M100 translator model is using {self.model.device}')
        self.logger.info(f'Supported languages: {[str(lang) for lang in self.supported_languages.keys()]}')

    def _translate_lines( self, lines ):
        encoded = self.tokenizer( lines, return_tensors="pt", padding=True, truncation=False )

        if encoded['input_ids'].shape[1] >= self.model.config.max_length:
            raise Exception(
                f"Input length ({encoded['input_ids'].shape[1]} tokens) is greater "
                f"than model maximum length ({self.model.config.max_length} tokens), "
                f"consider using sentence separation before translation."
            )

        encoded.to( self.model.device )
        generated_tokens = self.model.generate( **encoded, forced_bos_token_id=self.forced_bos_token_id )
        result = self.tokenizer.batch_decode( generated_tokens, skip_special_tokens=True )

        return result

    def _set_language_pair( self, source_language, target_language ):

        best_source_lang = find_closest_language( source_language, self.supported_languages.keys() )
        # Checking that best supported source language is not far away from the language that we asked to do
        if not best_source_lang:
            raise Exception(
                    f"Source language {source_language} is not supported. "
                    f"Supported languages are {[str(lang) for lang in self.supported_languages.keys()]}."
                )
        
        best_target_lang = find_closest_language( target_language, self.supported_languages.keys() )
        # Checking that best supported source language is not far away from the language that we asked to do
        if not best_target_lang:
            raise Exception(
                    f"Target language {target_language} is not supported. "
                    f"Supported languages are {[str(lang) for lang in self.supported_languages.keys()]}."
                )

        self.tokenizer.src_lang = self.supported_languages[best_source_lang]
        self.tokenizer.tgt_lang = self.supported_languages[best_target_lang]
        self.forced_bos_token_id = self.tokenizer.get_lang_id( self.supported_languages[best_target_lang] )

        return
