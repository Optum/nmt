import torch.cuda
import langcodes
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language

class TranslatorMBart50( TranslatorBase ):
    def __init__( self, eng_src_only = False, device = None,
        max_batch_lines = 8, max_batch_chars = 2000, max_file_lines = 60,
        verbose = False ):

        super(TranslatorMBart50, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        self.logger.info(f'Cuda is avaliable: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            self.logger.info(f'Number of avaliable GPUs: {torch.cuda.device_count()}')

        model_name = 'facebook/mbart-large-50-one-to-many-mmt' if eng_src_only else 'facebook/mbart-large-50-many-to-many-mmt'

        with Timer( f'MBart50 model initialization', self.logger ):
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

        self.model.eval()
        if device:
            self.model.to( device )
        elif torch.cuda.is_available():
            self.model.to('cuda')

        self.english_source_only = eng_src_only
        self.supported_languages = { langcodes.Language.get( lang ): lang for lang in self.tokenizer.lang_code_to_id.keys() }

        self.logger.info(f'MBart50 translator model is using {self.model.device}')
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

        if self.english_source_only:
            best_source_lang = langcodes.Language.get('en_XX')
            if not find_closest_language( best_source_lang, [source_language] ):
                raise Exception(
                        f"Source language {source_language} is not supported. "
                        f"The only supported source languages is {str(best_source_lang)}."
                    )
            self.tokenizer.src_lang = 'en_XX'
        else:
            best_source_lang = find_closest_language( source_language, self.supported_languages.keys() )
            # Checking that best supported source language is not far away from the language that we asked to do
            if not best_source_lang:
                raise Exception(
                        f"Source language {source_language} is not supported. "
                        f"Supported languages are {[str(lang) for lang in self.supported_languages.keys()]}."
                    )
            self.tokenizer.src_lang = self.supported_languages[best_source_lang]
        
        best_target_lang = find_closest_language( target_language, self.supported_languages.keys() )
        # Checking that best supported source language is not far away from the language that we asked to do
        if not best_target_lang:
            raise Exception(
                    f"Target language {target_language} is not supported. "
                    f"Supported languages are {[str(lang) for lang in self.supported_languages.keys()]}."
                )

        self.tokenizer.tgt_lang = self.supported_languages[best_target_lang]
        self.forced_bos_token_id = self.tokenizer.lang_code_to_id[self.supported_languages[best_target_lang]]

        return
