import os
import gc

import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv

from utils.utils import Timer, get_logger, log_cuda_info
from translator.create_translator import create_translator, Translator

# Loading environment variables for the project
load_dotenv()

# if some path is relative in config we consider it to be relative from the root project folder (and not cwd)
def _get_absolute_path( path ):
    if path:
        return os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', path ) )
    return path

# we store all the pathes as relative to the project root folder
def _get_realtive_path( path ):
    if path:
        return os.path.relpath( path, os.path.join( os.path.dirname( __file__ ), '..' ) )
    return path

def _translate_subset(  dataset, subset_name, translator,
    engine_name, engine_type,
    source_language, target_language,
    sentence_separation, reuse_previous_results ):

    source_file = dataset.get( subset_name, {} ).get( 'source', None )

    if not source_file:
        # We don't do anything if there is no file to translate
        return

    source_file = _get_absolute_path( source_file )

    source_file_folder = os.path.dirname( source_file )
    source_file_name, _ = os.path.splitext( os.path.basename( source_file ) )
    target_file = os.path.join( source_file_folder, f'{source_file_name}.{engine_name}.{target_language}' )

    with Timer( 'Translation timer', None ) as t:
        translation_status, translation_errors = translator.translate_file( source_file, target_file,
            source_language, target_language,
            sentence_separation, reuse_previous_results )
        translation_time = t.time

    dataset.setdefault( subset_name, {} ).setdefault( 'translation', [] ).append(
        {
            'name': engine_name,
            'type': engine_type,
            'file': _get_realtive_path( target_file ),
            'errors_count': len(translation_errors)
        }
    )

def _save_evaluation_config( set_name, evaluation_config_folder, datasets,
    source_language, target_language ):
    evaluation_config_folder = _get_absolute_path( evaluation_config_folder )
    if not os.path.isdir( evaluation_config_folder ):
        os.makedirs( evaluation_config_folder, exist_ok=False )
    evaluation_config_file = os.path.join( evaluation_config_folder, f'{set_name}.yaml' )
    config_to_save = {
        'source_language': source_language,
        'target_language': target_language,
        'datasets': datasets
    }
    OmegaConf.save( config = OmegaConf.create( config_to_save ), f = evaluation_config_file )

@hydra.main(config_path="../configs", config_name="02_translate_datasets")
def translate_datasets(cfg) -> None:
    logger = get_logger( 'translate_datasets', cfg.get( 'verbose', True ) )

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    set_name = hydra_cfg.runtime.choices.get( 'datasets_to_translate','' )
    logger.info(f"Datasets set used: {set_name}")

    source_language = cfg.get( 'source_language', 'en' )
    target_language = cfg.get( 'target_language', 'es' )

    sentence_separation = cfg.get( 'sentence_separation', False )
    reuse_previous_results = cfg.get( 'reuse_previous_results', True )

    # We want to update datasets information with link to newly created translations
    # That's why we want to transfer it from read-only config to modifyable list straight away
    datasets = OmegaConf.to_container(
        cfg.get( 'datasets_to_translate', {} ).get( 'datasets', [] ),
        resolve=True
    )

    source_language = cfg.get( 'datasets_to_translate', {} ).get( 'source_language', 'en' )
    target_language = cfg.get( 'datasets_to_translate', {} ).get( 'target_language', 'es' )

    engines = cfg.get( 'translation_engines', [] )

    translate_test = cfg.get( 'translate_subsets', {} ).get( 'test', True )
    translate_dev = cfg.get( 'translate_subsets', {} ).get( 'dev', False )
    translate_train = cfg.get( 'translate_subsets', {} ).get( 'train', False )

    for engine in engines:
        engine_class = engine.get( 'class', '' )
        engine_name = engine.get( 'name', '' )
        engine_type = engine.get( 'type', '' )
        engine_settings = engine.get( 'settings', {} )

        logger.info( f'----------Translating data with {engine_name} translator----------')

        translator = create_translator( Translator[engine_class], verbose = cfg.get( 'verbose', True ), **engine_settings )

        for dataset in datasets:
            dataset_name = dataset.get( 'name', '' )

            if translate_test:
                with Timer( f'Translating test subset of {dataset_name}', logger ):
                    _translate_subset( dataset, 'test', translator,
                        engine_name, engine_type,
                        source_language, target_language,
                        sentence_separation, reuse_previous_results )

            if translate_dev:
                with Timer( f'Translating dev subset of {dataset_name}', logger ):
                    _translate_subset( dataset, 'dev', translator,
                        engine_name, engine_type,
                        source_language, target_language,
                        sentence_separation, reuse_previous_results )
            
            if translate_train:
                with Timer( f'Translating train subset of {dataset_name}', logger ):
                    _translate_subset( dataset, 'train', translator,
                        engine_name, engine_type,
                        source_language, target_language,
                        sentence_separation, reuse_previous_results )

        # Sometimes even when object is deleted Cuda can still hold some GPU memory
        # Thats why we delete translator explicitly and run garbage collection
        del translator
        gc.collect()
        log_cuda_info( logger )

    create_evaluation_config = cfg.get( 'create_evaluation_config', False )
    if create_evaluation_config:
        evaluation_config_folder = cfg.get( 'evaluation_config_folder', '' )
        _save_evaluation_config( set_name, evaluation_config_folder, datasets,
            source_language, target_language )

if __name__ == "__main__":
    translate_datasets()