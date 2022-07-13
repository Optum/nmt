import langcodes
from google.cloud import translate
from google.auth import default, load_credentials_from_file

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language

class TranslatorGoogle( TranslatorBase ):
    def __init__( self, credentials_file = None,
        max_batch_lines = 30, max_batch_chars = 5000, max_file_lines = 300,
        verbose = False ):

        super(TranslatorGoogle, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        if credentials_file:
            credentials, project_id = load_credentials_from_file(credentials_file)
        else:
            # If no cred file is provided we assume it will come from default environment variable GOOGLE_APPLICATION_CREDENTIALS
            credentials, project_id = default()

        self.client = translate.TranslationServiceClient(credentials=credentials)
        location = "global"
        self.parent = f"projects/{project_id}/locations/{location}"

        languages = self.client.get_supported_languages(parent=self.parent)

        self.supported_source_languages = { langcodes.Language.get( lang.language_code ): lang.language_code for lang in languages.languages if lang.support_source }
        self.supported_target_languages = { langcodes.Language.get( lang.language_code ): lang.language_code for lang in languages.languages if lang.support_target }

        self.logger.info(f'Google MT supported source languages: {[str(lang) for lang in self.supported_source_languages.keys()]}')
        self.logger.info(f'Google MT supported target languages: {[str(lang) for lang in self.supported_target_languages.keys()]}')

    def _translate_lines( self, lines ):
        response = self.client.translate_text(
            parent=self.parent,
            contents=lines,
            source_language_code=self.source_language_code,
            target_language_code=self.target_language_code,
            mime_type="text/plain"
        )
        result = [translation.translated_text for translation in response.translations]
        return result

    def _set_language_pair( self, source_language, target_language ):
        best_source_lang = find_closest_language( source_language, self.supported_source_languages.keys() )
        # Checking that best supported source language is not far away from the language that we asked to do
        if not best_source_lang:
            raise Exception(
                    f"Source language {source_language} is not supported. "
                    f"Supported languages are {[str(lang) for lang in self.supported_source_languages.keys()]}."
                )
        
        best_target_lang = find_closest_language( target_language, self.supported_target_languages.keys() )
        # Checking that best supported source language is not far away from the language that we asked to do
        if not best_target_lang:
            raise Exception(
                    f"Target language {target_language} is not supported. "
                    f"Supported languages are {[str(lang) for lang in self.supported_target_languages.keys()]}."
                )

        self.source_language_code = self.supported_source_languages[best_source_lang]
        self.target_language_code = self.supported_target_languages[best_target_lang]

        return
