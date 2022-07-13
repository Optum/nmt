import abc
import os
import itertools
import unicodedata
import re
from enum import Enum, auto

import stanza
import langcodes
from tqdm import tqdm

from utils.utils import get_logger, buf_count_newlines_gen

class TranslationStatus(Enum):
    Translated = auto()
    Empty = auto()
    Failed = auto()
    PreviousUsed = auto()
    PartiallyFailed = auto()

class TranslatorBase(abc.ABC):
    def __init__( self, max_batch_lines, max_batch_chars, max_file_lines, verbose ) -> None:
        self.max_batch_lines = max_batch_lines
        self.max_batch_chars = max_batch_chars

        self.max_file_lines = max_file_lines

        self.verbose = verbose
        self.logger = get_logger( self.__class__.__name__, verbose )

        self.nlp_processor = {}

        self.__init_unicode_sets()

    def translate_file( self, source_file, target_file, source_language, target_language, sentence_separation = False, reuse_previous_output = True):
        try:
            self._set_language_pair( source_language, target_language )
        except Exception as x:
            self.logger.error(x)
            return TranslationStatus.Failed, repr(x)

        self.logger.info(f'Translating file: {source_file}')
        self.logger.info(f'Target file: {target_file}')

        # If we have previous translations, then let's load them to memory
        previous_translations = []
        if reuse_previous_output and os.path.isfile(target_file):
            with open( target_file, 'r', encoding='utf8' ) as input:
                previous_translations = [line.strip() for line in input]
        previous_translations_index = 0
        # This part is not very pretty, from perf point of view we need to read previous translations from file simlatneously with input
        # Bit it will raise a lot of edge cases, especially if we want to store translations to the same file that we got previous results from
        # So for simplicity sake let's live with this solution for now

        status = set()
        errors = []

        lines_to_translate = buf_count_newlines_gen( source_file )

        with open( source_file, 'r', encoding='utf8' ) as input, open( target_file, 'w', encoding='utf8' ) as output:

            progress_bar = tqdm( total=lines_to_translate, disable=(not self.verbose), desc='Translating file' )

            while True:
                next_n_lines = list(itertools.islice(input, self.max_file_lines))
                if not next_n_lines:
                    break

                translations, chunk_status, chunk_errors = self.translate_strings(
                    next_n_lines, source_language, target_language, sentence_separation, 
                    previous_translations[previous_translations_index:previous_translations_index+len(next_n_lines)])

                previous_translations_index += len(next_n_lines)

                status.update( chunk_status )
                errors.extend( [error for error in chunk_errors if error != ''] )

                output.write( '\n'.join(translations) + '\n' )

                progress_bar.update(len(next_n_lines))

            progress_bar.close()

        if len(status) == 1:
            final_status = status.pop()
        elif TranslationStatus.Failed in status:
            if TranslationStatus.Translated not in status:
                final_status = TranslationStatus.Failed
            else:
                final_status = TranslationStatus.PartiallyFailed
        else:
            final_status = TranslationStatus.Translated

        return final_status, errors

    def translate_strings( self, input_strings, source_language, target_language, sentence_separation = False, previous_translations = None ):
        """
        Translate given list of strings
        Arguments:
            input_strings: list of strings to translate
            source_language: language of input strings
            target_language: to what language to translate strings
            previous_translations: if not None, then it should be list parallel to input_strings. If present, only empty strings from previous_translations will be translated
        Returns:
            Three synchronized lists
            translated_strings: translation of the input strings
            status: for each string is translation was successful or not
            error: if something broke, then what the problem was
        """
        
        try:
            self._set_language_pair( source_language, target_language )
        except Exception as x:
            self.logger.error(x)
            result = ['']*len(input_strings)
            status = [TranslationStatus.Failed]*len(input_strings)
            errors = [repr(x)]*len(input_strings)
            return result, status, errors

        if not previous_translations:
            previous_translations = []
        if len(previous_translations) > len( input_strings ):
            previous_translations = previous_translations[:len(input_strings)]

        translated_result = ['']*len(input_strings)
        translated_error = ['']*len(input_strings)
        translated_status = [TranslationStatus.Empty]*len(input_strings)

        strings_to_translate = []
        translation_indices = []

        for index, (string, prev_translations_string) in enumerate( itertools.zip_longest( input_strings, previous_translations, fillvalue='' ) ):
            if prev_translations_string != '':
                translated_result[index] = prev_translations_string
                translated_status[index] = TranslationStatus.PreviousUsed
                translated_error[index] = ''
            else:
                cleared_string = self._cleanup_text( string )
                if cleared_string == '':
                    translated_result[index] = ''
                    translated_status[index] = TranslationStatus.Empty
                    translated_error[index] = ''
                else:
                    strings_to_translate.append( cleared_string )
                    translation_indices.append( index )
        
        # Sentence separation
        sentences = []
        sentence_indices = []

        if sentence_separation == True:    
            for strings in strings_to_translate:
                string_sentences = self._get_sentences( langcodes.Language.get( source_language ), strings )
                sentence_indices.append((len(sentences), len(string_sentences)))
                sentences.extend(string_sentences)
        else:
            for string_count in range(len(strings_to_translate)):
                sentence_indices.append((string_count,1))
            sentences = strings_to_translate
    
        curr_lines = []
        curr_char_count = 0
        result = []
        status = []
        errors = []
        for source_sentence in sentences:

            if len(curr_lines) + 1 > self.max_batch_lines or curr_char_count + len(source_sentence) > self.max_batch_chars:
                # If our batch is too big already, then process it
                if len(curr_lines) > 0:
                    # Let's process existing batch first
                    self._translate_batch(curr_lines, result, status, errors)
                    curr_lines = []
                    curr_char_count = 0

            curr_lines.append( source_sentence )
            curr_char_count += len( source_sentence )

        if len(curr_lines) > 0:
            # Process the last batch
            self._translate_batch(curr_lines, result, status, errors)
            curr_lines = []
            curr_char_count = 0

        # Join sentences in result
        for translation_index, (sentence_index, length) in zip( translation_indices, sentence_indices ):

            translated_string = ' '.join( result[sentence_index:sentence_index+length] ).strip()
            error_string = ' '.join( errors[sentence_index:sentence_index+length] ).strip()
            string_statuses = status[sentence_index:sentence_index+length]

            if TranslationStatus.Failed in string_statuses:
                if TranslationStatus.Translated not in string_statuses:
                    string_status = TranslationStatus.Failed
                else:
                    string_status = TranslationStatus.PartiallyFailed
            else:
                string_status = TranslationStatus.Translated

            translated_result[translation_index] = translated_string
            translated_status[translation_index] = string_status
            translated_error[translation_index] = error_string

        return translated_result, translated_status, translated_error

    def _translate_batch(self, curr_lines, result, status, errors):
        try:
            lines_result = self._translate_lines( curr_lines )
        except Exception as x:
            self.logger.error(x)
            result.extend( ['']*len(curr_lines) )
            status.extend( [TranslationStatus.Failed]*len(curr_lines) )
            errors.extend( [repr(x)]*len(curr_lines) )
        else:
            result.extend( lines_result )
            status.extend( [TranslationStatus.Translated]*len(curr_lines) )
            errors.extend( ['']*len(curr_lines) )

        return result

    def _get_nlp_processor( self, language ):
        lang_code = language.language
        if lang_code not in self.nlp_processor:
            self.nlp_processor[lang_code] = stanza.Pipeline( lang=lang_code, processors='tokenize' )
        
        return self.nlp_processor[lang_code]
        

    def _get_sentences(self, language, string):

        processor = self._get_nlp_processor( language )

        doc = processor(string)

        separated_sentences = []

        for sentence in doc.sentences:
            sentence_text = self._cleanup_text(sentence.text)
            if sentence_text != '':
                separated_sentences.append(sentence_text)

        return separated_sentences

    def _cleanup_text( self, line ):
        # Clean up text a little bit before translation

        # Replace all possible spaces with normal spaces
        line = re.sub( self.reg_spaces, ' ', line )
        # Remove all control characters
        line = re.sub( self.reg_control, '', line )

        # Remove space from begining and the end
        line = line.strip()

        return line

    def __init_unicode_sets( self ):
        self.punctuation, self.letters, self.numbers, self.spaces, self.control = \
            set(), set(), set(), set(), set()
        # We go through whole range of possible Unicode characters
        for i in range(0,0x110000):
            char = chr(i)
            category = unicodedata.category( char )
            # Punctuation is everything in P* category
            if( category.startswith('P') ):
                self.punctuation.add( char )
            # For our goal both letters (L*) and mark signs (M*)
            # will be considered as letters
            elif( category.startswith('L') or category.startswith('M') ):
                self.letters.add( char )
            # Nd and Nl goes to numbers (No is not exactly digits)
            elif( category == 'Nd' or category == 'Nl' ):
                self.numbers.add( char )
            # Z* goes to punctuation
            elif( category.startswith('Z') ):
                self.spaces.add( char )
            # We will need control (Cc) and format (Cf) characters a little bit later
            elif( category == 'Cc' or category == 'Cf' ):
                self.control.add( char )
        # TAB, CR and LF are in Cc category, but we will treat them as spaces
        self.spaces.update( ['\t','\r','\n'] )
        self.control.difference_update(  ['\t','\r','\n']  )

        self.reg_spaces = re.compile(f"[{''.join(self.spaces)}]")
        self.reg_control = re.compile(f"[{''.join(self.control)}]")

    @abc.abstractmethod
    def _set_language_pair( self, source_language, target_language ):
        """Model specific method to set language pair"""
        """We guarantee that languages in input will already be unified"""

    @abc.abstractmethod
    def _translate_lines( self, lines ):
        """Method to translate a batch of lines in one go"""
        return

