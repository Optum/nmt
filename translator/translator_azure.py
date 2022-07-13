import os

import langcodes
import requests

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language

class TranslatorAzure( TranslatorBase ):
    def __init__( self, azure_sub_key = None,
        max_batch_lines = 30, max_batch_chars = 5000, max_file_lines = 300,
        verbose = False ):

        super(TranslatorAzure, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        self.azure_api = 'https://api.cognitive.microsofttranslator.com/translate'

        try:
            lang_api = 'https://api.cognitive.microsofttranslator.com/languages'
            lang_params = {
                'api-version': '3.0',
                'scope': 'translation',
            }
            lang_response = requests.get( lang_api, params=lang_params )
            lang_response.raise_for_status()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise Exception(e)

        self.supported_languages = { langcodes.Language.get( lang ): lang for lang in lang_response.json()['translation'].keys() }

        self.azure_params = {
            'api-version': '3.0',
            'from': '',
            'to': ''
        }

        if not azure_sub_key:
            # If key is not provided it should be in the environment variable
            azure_sub_key = os.environ.get('AZURE_MT_KEY')

        self.azure_headers = {
            'Ocp-Apim-Subscription-Key': azure_sub_key,
            'Ocp-Apim-Subscription-Region': 'global',
            'Content-type': 'application/json'
        }

        self.logger.info(f'Azure MT supported languages: {[str(lang) for lang in self.supported_languages.keys()]}')

    def _translate_lines( self, lines ):
        body = [{'text': line} for line in lines]

        try:
            response = requests.post( self.azure_api, params=self.azure_params, headers=self.azure_headers, json=body )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise Exception(e)
        
        result = [translation['translations'][0]['text']  for translation in response.json()]
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

        self.azure_params['from'] = self.supported_languages[best_source_lang]
        self.azure_params['to'] = self.supported_languages[best_target_lang]

        return
