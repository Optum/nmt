from translator.translator_base import TranslatorBase

# Empty translator to test overall translator architecture
class TranslatorEmpty( TranslatorBase ):
    # required init params:
    # max_batch_lines - how many lines your engine can translate in one go
    # max_batch_chars - how many characters should be in one translation batch
    # max_file_lines - if we process text file,
    # how many lines we can read from file in one go to split to batches
    # (rule of thumb is 10*max_batch_lines)
    # verbose - do we want info output or only errors
    # You can add your own additional parameters here
    def __init__( self,
        max_batch_lines = 4, max_batch_chars = 1000, max_file_lines = 40,
        verbose = False ):

        super(TranslatorEmpty, self).__init__(
            max_batch_lines = max_batch_lines, max_batch_chars = max_batch_chars, max_file_lines = max_file_lines,
            verbose = verbose )

        # Your translator object should be created here

        self.logger.info(f'Created Empty translator engine')

    # Main function to translate lines
    # We guarantee that len(lines) <= max_batch_lines and
    # len( ''.join(lines) ) <= max_batch_chars
    def _translate_lines( self, lines ):
        result = [line for line in lines]
        return result

    # Setting source and target language for next calls of _translate_lines
    # Function is needed if we are using multilingual translation engines
    def _set_language_pair( self, source_language, target_language ):
        # Config your engine for the language pair
        return