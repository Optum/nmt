import os
import logging

import torch
import langcodes
from nemo.collections.nlp.models import MTEncDecModel

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language

class TranslatorNemo( TranslatorBase ):
    @staticmethod
    def get_pretrained_model_name( source_language, target_language, large_model = False ):
        # This is the list of Marian models that we used, 
        # transformers has much more pre-trained models.
        known_models = {
            False: {
                langcodes.Language.get( 'en' ): {
                    langcodes.Language.get( 'es' ) : "nmt_en_es_transformer12x2",
                    langcodes.Language.get( 'ru' ) : "nmt_en_ru_transformer6x6",
                    langcodes.Language.get( 'zh' ) : "nmt_en_zh_transformer6x6"
                }
            },
            True: {
                langcodes.Language.get( 'en' ): {
                    langcodes.Language.get( 'es' ) : "nmt_en_es_transformer24x6",
                    langcodes.Language.get( 'ru' ) : "nmt_en_ru_transformer24x6",
                    langcodes.Language.get( 'zh' ) : "nmt_en_zh_transformer24x6"
                }
            }
        }

        models_for_size = known_models[large_model]

        source_language = langcodes.Language.get( source_language )
        target_language = langcodes.Language.get( target_language )

        best_source_lang = find_closest_language( source_language, models_for_size.keys() )
        if not best_source_lang:
            # We don't know pre-trained model for this source language
            return None

        models_for_source = models_for_size[best_source_lang]
        best_target_lang = find_closest_language( target_language, models_for_source.keys() )
        if not best_target_lang:
            # We don't know pre-trained model for this target language
            return None

        return models_for_source[best_target_lang]

    # If we fine-tune NeMo model training process will create ckpt files each epoch
    # But to load this ckpt files we need to know original pretrained model config
    # But after the fine-tuning .nemo file will be created that
    # already contains pretrained config inside and can be loaded as is
    def __init__( self, pretrained_model = None, finetuned_file = None, device = None, beam_size = 4,
        max_batch_lines = 8, max_batch_chars = 2000, max_file_lines = 60,
        verbose = False ):

        super(TranslatorNemo, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        self.logger.info(f'Cuda is avaliable: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            self.logger.info(f'Number of avaliable GPUs: {torch.cuda.device_count()}')

        # Even WARNING level output is to much for NeMo
        logging.getLogger('nemo_logger').setLevel(logging.ERROR)

        if not pretrained_model and not finetuned_file:
            raise Exception(
                f"To load NeMo model either name of pre-trained model or finetuned file needs to be provided"
            )
        
        is_checkpoint = self._is_checkpoint_file( finetuned_file )

        if finetuned_file and is_checkpoint and not pretrained_model:
            raise Exception(
                f"Fine-tuned model file is pytorch checkpoint ({finetuned_file}). "
                f"We need to know pre-trained model name as well to load this file correctly."
            )

        if finetuned_file and not is_checkpoint and pretrained_model:
            self.logger.warning(
                f'Pre-trained model ({pretrained_model}) will be ignored, '
                'because fine-tuned file already have all necessary information' )

        with Timer( f'NeMo model config initialization', self.logger ):
            inference_config = self._get_model_inference_config( pretrained_model, finetuned_file, is_checkpoint, beam_size, device )

        with Timer( f'NeMo model initialization', self.logger ):
            self._load_model( pretrained_model, finetuned_file, is_checkpoint, inference_config, device )


        # Nemo models are bi-lingual (one language translates to one language)
        # We don't really need source and target language then after model is loaded
        # But it still will be good to check that we use the model for
        # the translation pair that it was intended for.

        # Unfortunately for us, NeMo not always keep proper languages in model config.
        if self.model.src_language:
            self.source_language = langcodes.Language.get( self.model.src_language )
        else:
            self.source_language = None

        if self.model.tgt_language and self.model.src_language != self.model.tgt_language:
            self.target_language = langcodes.Language.get( self.model.tgt_language )
        else:
            self.target_language = None


        self.logger.info(f'NeMo translator model is using {self.model.device}')
        self.logger.info(f'Model source language: {self.source_language}')
        self.logger.info(f'Model target language: {self.target_language}')

    def _translate_lines( self, lines ):
        src_lang = self.source_language.language if self.source_language else None
        trg_lang = self.target_language.language if self.target_language else None
        result = self.model.translate( lines, source_lang=src_lang, target_lang=trg_lang )

        return result

    def _set_language_pair( self, source_language, target_language ):
        # We don't need to specifically set source language for NeMo.
        # Let's just check that the language is supported
        if self.source_language:
            if not find_closest_language( source_language, [self.source_language] ):
                raise Exception(
                    f"Source language {source_language} is not supported. "
                    f"The only supported source language is {self.source_language}."
                )

        # We don't need to specifically set source language for NeMo.
        # Let's just check that the language is supported
        if self.target_language:
            if not find_closest_language( target_language, [self.target_language] ):
                raise Exception(
                    f"Target language {target_language} is not supported. "
                    f"The only supported target language is {self.target_language}."
                )

        return

    def _is_checkpoint_file( self, finetuned_file ):
        if finetuned_file:
            _, fine_tuned_extension = os.path.splitext( finetuned_file )

        return False

    def _get_model_inference_config( self, pretrained_name, finetuned_file, is_checkpoint, beam_size, device ):
        # By default NeMo config has a lot of random dropouts for better training
        # We need to disable them all for inference
        _NeMo_inference_config = {
            'encoder_tokenizer': { 'bpe_dropout': 0.0 },
            'decoder_tokenizer': { 'bpe_dropout': 0.0 },
            'encoder': {
                'embedding_dropout': 0.0,
                'ffn_dropout': 0.0,
                'attn_score_dropout': 0.0,
                'attn_layer_dropout': 0.0,
            },
            'decoder': {
                'embedding_dropout': 0.0,
                'ffn_dropout': 0.0,
                'attn_score_dropout': 0.0,
                'attn_layer_dropout': 0.0,
            },
            'head': { 'dropout': 0.0 },
        }

        if finetuned_file and not is_checkpoint:
            self.logger.info( f'Loading model config from file: {finetuned_file}' )
            model_config = MTEncDecModel.restore_from(
                finetuned_file, return_config = True, map_location = device )
        else:
            self.logger.info( f'Loading model config from pretrained model: {pretrained_name}' )
            model_config = MTEncDecModel.from_pretrained(
                pretrained_name, return_config = True, map_location = device  )

        model_config.merge_with( _NeMo_inference_config )
        
        model_config['beam_size'] = beam_size

        return model_config

    def _load_model( self, pretrained_name, finetuned_file, is_checkpoint, inference_config, device ):
        if finetuned_file and not is_checkpoint:
            self.logger.info( f'Loading model from file: {finetuned_file}' )
            self.model = MTEncDecModel.restore_from(
                finetuned_file, override_config_path = inference_config, map_location = device )
        else:
            self.logger.info( f'Restoring pretrained model: {pretrained_name}' )
            self.model = MTEncDecModel.from_pretrained(
                pretrained_name, override_config_path = inference_config, map_location = device )
            if finetuned_file and is_checkpoint:
                self.logger.info( f'Restoring weights from checkpoint: {finetuned_file}' )
                checkpoint = torch.load( finetuned_file, map_location=device )
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                self.model.load_state_dict(checkpoint, strict=True)

        self.model.decoder.return_mems = True

        self.model.eval()
