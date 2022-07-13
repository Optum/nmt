import os

import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv

from utils.utils import Timer, get_logger
from datasets.opus_dataset import OpusDatasetCreator, DatasetProperties

# Loading environment variables for the project
load_dotenv()

# if some path is relative in config we consider it to be relative from the project root folder (and not cwd)
def _get_absolute_path( path ):
    if path:
        return os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..', path ) )
    return path

# we store all the pathes as relative to the project root folder
def _get_realtive_path( path ):
    if path:
        return os.path.relpath( path, os.path.join( os.path.dirname( __file__ ), '..' ) )
    return path

def _get_dataset_properties( dataset_config ):
    return DatasetProperties(
        size = dataset_config.get( 'size', 0 ),
        no_duplicates = dataset_config.get( 'no_duplicates', True ),
        min_str_length = dataset_config.get( 'min_str_length', 0 )
    )

def _create_config_info( dataset_name, dataset_type, test_info, dev_info, train_info ):
    return {
        'name': dataset_name,
        'type': dataset_type,
        'test': {
            'source': _get_realtive_path( test_info.source_file ),
            'target': _get_realtive_path( test_info.target_file ),
            'size': test_info.size
        },
        'dev': {
            'source': _get_realtive_path( dev_info.source_file ),
            'target': _get_realtive_path( dev_info.target_file ),
            'size': dev_info.size
        },
        'train': {
            'source': _get_realtive_path( train_info.source_file ),
            'target': _get_realtive_path( train_info.target_file ),
            'size': train_info.size
        }
    }

def _save_translator_config( set_name, translator_config_folder, dataset_config_list,
    source_language, target_language ):
    translator_config_folder = _get_absolute_path( translator_config_folder )
    if not os.path.isdir( translator_config_folder ):
        os.makedirs( translator_config_folder, exist_ok=False )
    translator_config_file = os.path.join( translator_config_folder, f'{set_name}.yaml' )
    config_to_save = {
        'source_language': source_language,
        'target_language': target_language,
        'datasets': dataset_config_list
    }
    OmegaConf.save( config = OmegaConf.create( config_to_save ), f = translator_config_file )

@hydra.main(config_path="../configs", config_name="01_download_datasets")
def download_datasets(cfg) -> None:
    logger = get_logger( 'download_datasets', cfg.get( 'verbose', True ) )

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    set_name = hydra_cfg.runtime.choices.get( 'datasets_to_download','' )
    logger.info(f"Datasets set used: {set_name}")

    dataset_folder = _get_absolute_path( cfg.get( 'datasets_folder' ) )
    temp_folder = _get_absolute_path( cfg.get( 'temp_folder', None ) )

    datase_creator = OpusDatasetCreator(
        datasets_folder = dataset_folder,
        temp_folder = temp_folder,
        verbose = cfg.get( 'verbose', True ),
        random_seed = cfg.get( 'random_seed', None ),
        reuse_downloaded_files = cfg.get( 'reuse_downloaded_files', False ),
    )

    source_language = cfg.get( 'source_language', 'en' )
    target_language = cfg.get( 'target_language', 'es' )

    datasets = cfg.get( 'datasets_to_download', {} ).get( 'datasets', [] )

    source_language = cfg.get( 'datasets_to_download', {} ).get( 'source_language', 'en' )
    target_language = cfg.get( 'datasets_to_download', {} ).get( 'target_language', 'es' )

    dataset_config_list = []

    for dataset in datasets:
        dataset_name = dataset.get( 'name', '' )
        dataset_type = dataset.get( 'type', '' )
        test_properties = _get_dataset_properties( dataset.get( 'test', {} ) )
        dev_properties = _get_dataset_properties( dataset.get( 'dev', {} ) )
        train_properties = _get_dataset_properties( dataset.get( 'train', {} ) )

        logger.info( f'----------Preparing dataset {dataset_name}----------')

        test_info, dev_info, train_info = datase_creator.create_dataset(
            dataset_name = dataset_name,
            source_language = source_language,
            target_language = target_language,
            test_properties = test_properties,
            dev_properties = dev_properties,
            train_properties = train_properties
        )

        dataset_config_list.append( _create_config_info( dataset_name, dataset_type,
            test_info, dev_info, train_info ) )

    create_translator_config = cfg.get( 'create_translator_config', False )
    if create_translator_config:
        translator_config_folder = cfg.get( 'translator_config_folder', '' )
        _save_translator_config( set_name, translator_config_folder, dataset_config_list,
            source_language, target_language )

if __name__ == "__main__":
    download_datasets()