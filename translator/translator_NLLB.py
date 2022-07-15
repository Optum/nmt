import torch.cuda
import langcodes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from enum import Enum

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language

class NLLBSize(Enum):
    Distilled_600M = "facebook/nllb-200-distilled-600M"
    Distilled_13B = "facebook/nllb-200-distilled-1.3B"
    Full_13B = "facebook/nllb-200-1.3B"
    Full_33B = "facebook/nllb-200-3.3B"

class TranslatorNLLB( TranslatorBase ):
    def __init__( self, model_size = 'Distilled_600M', device = None,
        max_batch_lines = 4, max_batch_chars = 2000, max_file_lines = 60,
        verbose = False ):

        super(TranslatorNLLB, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        self.logger.info(f'Cuda is avaliable: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            self.logger.info(f'Number of avaliable GPUs: {torch.cuda.device_count()}')

        model_name = NLLBSize[model_size].value

        with Timer( f'NLLB model initialization', self.logger ):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        self.model.config.max_length = self.tokenizer.model_max_length

        self.model.eval()
        if device:
            self.model.to( device )
        elif torch.cuda.is_available():
            self.model.to('cuda')

        self.supported_languages = { langcodes.Language.get( lang ): lang for lang in self.tokenizer.lang_code_to_id.keys() }

        self.logger.info(f'NLLB translator model is using {self.model.device}')
        self.logger.info(f'Supported languages: {[str(lang) for lang in self.supported_languages.keys()]}')

    def _translate_lines( self, lines ):
        encoded = self.tokenizer( lines, return_tensors="pt", padding=True, truncation=False )

        if encoded['input_ids'].shape[1] >= self.tokenizer.model_max_length:
            raise Exception(
                f"Input length ({encoded['input_ids'].shape[1]} tokens) is greater "
                f"than model maximum length ({self.tokenizer.model_max_length} tokens), "
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
