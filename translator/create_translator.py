from importlib import import_module
from enum import Enum

class Translator(Enum):
    Empty = 'translator.translator_empty.TranslatorEmpty'
    Azure = 'translator.translator_azure.TranslatorAzure'
    Google = 'translator.translator_google.TranslatorGoogle'
    Marian = 'translator.translator_marian.TranslatorMarian'
    Nemo = 'translator.translator_nemo.TranslatorNemo'
    M2M100 = 'translator.translator_M2M100.TranslatorM2M100'
    MBart50 = 'translator.translator_MBart50.TranslatorMBart50'
    NLLB = 'translator.translator_NLLB.TranslatorNLLB'

def create_translator( translator: Translator, verbose = False, **kwargs ):
    try:
        class_str = translator.value
        module_path, class_name = class_str.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)( verbose = verbose, **kwargs )
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)
