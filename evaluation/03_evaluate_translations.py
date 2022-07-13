import os

import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv

from utils.utils import Timer, get_logger
from scorer.dataset_scorer import DatasetScorer
from scorer.create_scorer import Scorer

# Loading environment variables for the project
load_dotenv()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

def _evaluate_subset( dataset, subset_name, global_scorer, logger ):

    source_file = dataset.get( subset_name, {} ).get( 'source', None )
    reference_file = dataset.get( subset_name, {} ).get( 'target', None )
    if not source_file or not reference_file:
        # We don't do anything if there is no file to evaluate
        return

    source_file = _get_absolute_path( source_file )
    reference_file = _get_absolute_path( reference_file )

    global_scorer.set_source_and_reference( source_file, reference_file )

    for translation in dataset.get( subset_name, {} ).get( 'translation', [] ):
        translation_file = translation.get( 'file', None )
        engine_name = translation.get( 'name', None )
        if not translation_file:
            # We don't do anything if there is no file to evaluate
            continue
        translation_file = _get_absolute_path( translation_file )
        with Timer( f'Evaluating translation for {engine_name}', logger ):
            scores = global_scorer.score_translations( translation_file )

        translation['score'] = scores

def _save_evaluation_results_yaml( set_name, results_folder, datasets,
    source_language, target_language ):
    results_folder = _get_absolute_path( results_folder )
    if not os.path.isdir( results_folder ):
        os.makedirs( results_folder, exist_ok=False )
    results_file = os.path.join( results_folder, f'{set_name}.yaml' )
    config_to_save = {
        'source_language': source_language,
        'target_language': target_language,
        'datasets': datasets
    }
    OmegaConf.save( config = OmegaConf.create( config_to_save ), f = results_file )

def _save_evaluation_results_tsv( set_name, results_folder, datasets,
    source_language, target_language ):
    results_folder = _get_absolute_path( results_folder )
    if not os.path.isdir( results_folder ):
        os.makedirs( results_folder, exist_ok=False )
    results_file = os.path.join( results_folder, f'{set_name}.tsv' )

    # Trying to determine what scores we have calculated
    scorers = set()

    for dataset in datasets:
        for subset_type in ['test','dev','train']:
            subset = dataset.get( subset_type, {} )
            translations = subset.get( 'translation', [] )
            for translation in translations:
                scores = translation.get( 'score', {} )
                scorers.update( scores.keys() )

    scorers = sorted( list( scorers) )

    with open( results_file, 'w', encoding='utf8' ) as tsv:
        header = '\t'.join( [
            'dataset_name',
            'dataset_type',
            'subset_type',
            'size',
            'source_language',
            'target_language',
            'engine_name',
            'engine_type',
            'errors_count'] + scorers ) + '\n'
        tsv.write( header )

        for dataset in datasets:
            dataset_name = dataset.get( 'name', '' )
            dataset_type = dataset.get( 'type', '' )
            for subset_type in ['test','dev','train']:
                subset = dataset.get( subset_type, {} )
                size = subset.get( 'size', 0 )
                translations = subset.get( 'translation', [] )
                for translation in translations:
                    engine_name = translation.get( 'name', '' )
                    engine_type = translation.get( 'type', '' )
                    errors_count = translation.get( 'errors_count', 0.0 )
                    translation_info = '\t'.join( [
                        dataset_name,
                        dataset_type,
                        subset_type,
                        str(size),
                        source_language,
                        target_language,
                        engine_name,
                        engine_type,
                        str(errors_count)
                    ] )
                    scores = translation.get( 'score', {} )
                    for scorer in scorers:
                        score = scores.get( scorer, '' )
                        translation_info += f'\t{score}'
                    tsv.write( translation_info + '\n' )


@hydra.main(config_path="../configs", config_name="03_evaluate_translations")
def evaluate_translations(cfg) -> None:
    logger = get_logger( 'evaluate_translations', cfg.get( 'verbose', True ) )

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    set_name = hydra_cfg.runtime.choices.get( 'datasets_to_evaluate','' )
    logger.info(f"Datasets set used: {set_name}")

    # We want to update datasets information with evaluation scores
    # That's why we want to transfer it from read-only config to modifyable list straight away
    datasets = OmegaConf.to_container(
        cfg.get( 'datasets_to_evaluate', {} ).get( 'datasets', [] ),
        resolve=True
    )

    source_language = cfg.get( 'datasets_to_evaluate', {} ).get( 'source_language', 'en' )
    target_language = cfg.get( 'datasets_to_evaluate', {} ).get( 'target_language', 'es' )

    scorers = cfg.get( 'translation_scorers', [] )

    scorers_settings = {
        Scorer[scorer.get('class', '')]: scorer.get('settings', {}) for scorer in scorers
    }

    evaluate_test = cfg.get( 'evaluation_subsets', {} ).get( 'test', True )
    evaluate_dev = cfg.get( 'evaluation_subsets', {} ).get( 'dev', False )
    evaluate_train = cfg.get( 'evaluation_subsets', {} ).get( 'train', False )

    global_scorer = DatasetScorer(
        source_language = source_language,
        target_language = target_language,
        required_scorers = scorers_settings.keys(),
        verbose = cfg.get( 'verbose', True ),
        scorer_configs = scorers_settings )

    for dataset in datasets:
        dataset_name = dataset.get( 'name', '' )

        if evaluate_test:
            with Timer( f'Evaluating test subset of {dataset_name}', logger ):
                _evaluate_subset( dataset, 'test', global_scorer, logger  )

        if evaluate_dev:
            with Timer( f'Evaluating dev subset of {dataset_name}', logger ):
                _evaluate_subset( dataset, 'dev', global_scorer, logger  )

        if evaluate_train:
            with Timer( f'Evaluating train subset of {dataset_name}', logger ):
                _evaluate_subset( dataset, 'train', global_scorer, logger  )

    create_yaml_evaluation_results = cfg.get( 'create_yaml_evaluation_results', False )
    if create_yaml_evaluation_results:
        yaml_evaluation_results_folder = cfg.get( 'yaml_evaluation_results_folder', '' )
        _save_evaluation_results_yaml( set_name, yaml_evaluation_results_folder, datasets,
            source_language, target_language )

    create_tsv_evaluation_results = cfg.get( 'create_tsv_evaluation_results', False )
    if create_yaml_evaluation_results:
        tsv_evaluation_results_folder = cfg.get( 'tsv_evaluation_results_folder', '' )
        _save_evaluation_results_tsv( set_name, tsv_evaluation_results_folder, datasets,
            source_language, target_language )

if __name__ == "__main__":
    evaluate_translations()