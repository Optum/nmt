import os
import zipfile
import random
import shutil
from tempfile import TemporaryDirectory
from datetime import datetime

import requests
import langcodes
from tqdm import tqdm

from utils.utils import get_logger, Timer, find_closest_language, buf_count_newlines_gen

class DatasetProperties:
    def __init__(self, size = 1000, no_duplicates = True, min_str_length = 0 ) -> None:
        self._size = size
        self._no_duplicates = no_duplicates
        self._min_str_length = min_str_length

    @property
    def size(self):
        return self._size

    @property
    def no_duplicates(self):
        return self._no_duplicates

    @property
    def min_str_length(self):
        return self._min_str_length

class DatasetInfo:
    def __init__(self, source_file, target_file, size = None ) -> None:
        self._source_file = source_file
        self._target_file = target_file
        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def source_file(self):
        return self._source_file

    @property
    def target_file(self):
        return self._target_file

class OpusDatasetCreator:
    def __init__( self, datasets_folder, temp_folder = None, verbose = True, random_seed = None,
        reuse_downloaded_files = False ) -> None:
        self.opus_api = 'https://opus.nlpl.eu/opusapi/'

        self.logger = get_logger( self.__class__.__name__, verbose )

        if random_seed:
            self.random_seed = random_seed
        else:
            self.random_seed = datetime.now().timestamp()

        self.reuse_downloaded_files = reuse_downloaded_files

        self.datasets_folder = datasets_folder

        if not temp_folder:
            # Creating folder to keep temp objects and keeping link to the object to delete when dataset object is deleted
            self.temp_folder_object = TemporaryDirectory()
            self.temp_folder = self.temp_folder_object.name
        else:
            # We have temp folder from the user
            self.temp_folder_object = None
            self.temp_folder = temp_folder

        self.logger.info( f'Datasets folder: {self.datasets_folder}' )
        self.logger.info( f'Temp folder: {self.temp_folder}' )
        self.logger.info( f'Random seed: {self.random_seed}' )

    def create_dataset( self, dataset_name, source_language, target_language,
        test_properties = DatasetProperties(), dev_properties = DatasetProperties(), train_properties = DatasetProperties() ):

        with Timer( f'Downloading dataset', self.logger ):
            dataset_src_lang = self._get_dataset_src_lang( dataset_name, source_language )
            dataset_trg_lang = self._get_dataset_trg_lang( dataset_name, dataset_src_lang, target_language )

            dataset_zip = self._download_zip( dataset_name, dataset_src_lang, dataset_trg_lang )

        with Timer( f'Extracting files', self.logger ):
            dataset_info = self._extract_files( dataset_zip, dataset_name, dataset_src_lang, dataset_trg_lang )

        with Timer( f'Creating test set', self.logger ):
            src_skip_set = set()
            trg_skip_set = set()

            test_info, dataset_info = self._sample_dataset( dataset_info, test_properties, src_skip_set, trg_skip_set )

            test_info_final = self._copy_dataset( test_info, dataset_name, source_language, target_language, 'test' )

        with Timer( f'Creating dev set', self.logger ):
            # We don't want lines from test to present in dev set
            src_skip_set.update( self._create_strings_set( test_info.source_file ) )
            trg_skip_set.update( self._create_strings_set( test_info.target_file ) )

            dev_info, dataset_info = self._sample_dataset( dataset_info, dev_properties, src_skip_set, trg_skip_set )

            dev_info_final = self._copy_dataset( dev_info, dataset_name, source_language, target_language, 'dev' )
        
        with Timer( f'Creating train set', self.logger ):
            # We don't want lines from test or dev to present in train set
            src_skip_set.update( self._create_strings_set( dev_info.source_file ) )
            trg_skip_set.update( self._create_strings_set( dev_info.target_file ) )

            train_info, dataset_info = self._sample_dataset( dataset_info, train_properties, src_skip_set, trg_skip_set )

            train_info_final = self._copy_dataset( train_info, dataset_name, source_language, target_language, 'train' )

        return test_info_final, dev_info_final, train_info_final

    def _get_dataset_src_lang( self, dataset_name, source_language ):
        try:
            request_params = {
                'languages': 'True',
                'corpus': dataset_name,
            }
            lang_response = requests.get( self.opus_api, params=request_params )
            lang_response.raise_for_status()

            avaliable_languages = lang_response.json()['languages']
        except requests.exceptions.RequestException as e:
            raise Exception(e)

        if len(avaliable_languages) == 0:
            raise Exception(
                f"No supported languages found for the dataset {dataset_name}. "
                f"Please check that dataset name is correct."
            )

        best_source_lang = find_closest_language( source_language, avaliable_languages )

        if not best_source_lang:
            raise Exception(
                f"Source language {source_language} is not supported for the dataset {dataset_name}. "
                f"Supported languages are {[str(lang) for lang in avaliable_languages]}."
            )

        self.logger.info( f'Found source language {best_source_lang} for dataset {dataset_name} (requested {source_language})')

        return best_source_lang

    def _get_dataset_trg_lang( self, dataset_name, dataset_src_lang, target_language ):
        try:
            request_params = {
                'languages': 'True',
                'source': dataset_src_lang,
                'corpus': dataset_name,
            }
            lang_response = requests.get( self.opus_api, params=request_params )
            lang_response.raise_for_status()

            avaliable_languages = lang_response.json()['languages']
        except requests.exceptions.RequestException as e:
            raise Exception(e)

        if len(avaliable_languages) == 0:
            raise Exception(
                f"No supported target languages found for the dataset {dataset_name} and the source language {dataset_src_lang}. "
                f"Please check that dataset name is correct."
            )

        best_target_lang = find_closest_language( target_language, avaliable_languages )

        if not best_target_lang:
            raise Exception(
                f"Target language {target_language} is not supported for the dataset {dataset_name} and the source language {dataset_src_lang}. "
                f"Supported target languages are {[str(lang) for lang in avaliable_languages]}."
            )

        self.logger.info( f'Found target language {best_target_lang} for dataset {dataset_name} (requested {target_language})')

        return best_target_lang

    def _download_zip( self, dataset_name, dataset_src_lang, dataset_trg_lang ):
        try:
            request_params = {
                'source': dataset_src_lang,
                'target': dataset_trg_lang,
                'corpus': dataset_name,
                'preprocessing': 'moses',
                'version': 'latest'
            }
            dataset_response = requests.get( self.opus_api, params=request_params )
            dataset_response.raise_for_status()

            dataset_url = dataset_response.json()['corpora'][0]['url']
        except requests.exceptions.RequestException as e:
            raise Exception(e)

        self.logger.info( f'Download url: {dataset_url}' )

        dataset_temp_folder = os.path.join( self.temp_folder, f'{dataset_src_lang}.{dataset_trg_lang}', dataset_name )
        if not os.path.isdir( dataset_temp_folder ):
            os.makedirs( dataset_temp_folder, exist_ok=False )
        self.logger.info( f'Dataset temp folder: {dataset_temp_folder}' )

        zip_filename = dataset_url.split('/')[-1]
        dataset_zip = os.path.join( dataset_temp_folder, zip_filename )
        self.logger.info( f'Dataset zip file name: {dataset_zip}' )

        if self.reuse_downloaded_files and os.path.isfile( dataset_zip ):
            self.logger.info( f'Using previously downloaded file: {dataset_zip}' )
            return dataset_zip

        size_response = requests.head(dataset_url)
        zip_size = int(size_response.headers['Content-Length'])
        self.logger.info( f'Zip file size: {1.0*zip_size/(1024*1024):.2f}MB' )

        chunk_size = 5*1024*1024 # size of 1 chunk 5 MB

        try:
            with requests.get(dataset_url, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                with open(dataset_zip, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size = chunk_size), total = 1 + zip_size // chunk_size, desc='Download dataset'):
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            raise Exception(e)

        return dataset_zip

    def _extract_files( self, dataset_zip, dataset_name, dataset_src_lang, dataset_trg_lang ):
        source_file_name = f'{dataset_name}.{dataset_src_lang}-{dataset_trg_lang}.{dataset_src_lang}'
        target_file_name = f'{dataset_name}.{dataset_src_lang}-{dataset_trg_lang}.{dataset_trg_lang}'

        dataset_temp_dir = os.path.dirname( dataset_zip )

        source_file_path = os.path.join( dataset_temp_dir, source_file_name )
        target_file_path = os.path.join( dataset_temp_dir, target_file_name )

        if self.reuse_downloaded_files and os.path.isfile( source_file_path ) and os.path.isfile( target_file_path ):
            self.logger.info( f'Using previously extracted files: {source_file_path} and {target_file_path}' )
            return DatasetInfo( source_file_path, target_file_path )

        
        with zipfile.ZipFile( dataset_zip, 'r' ) as zip_ref:
            with Timer( f'Extracting source file {source_file_name}', self.logger ):
                source_file = zip_ref.extract( source_file_name, path = dataset_temp_dir )
            with Timer( f'Extracting target file {target_file_name}', self.logger ):
                target_file = zip_ref.extract( target_file_name, path = dataset_temp_dir )
        
        return DatasetInfo( source_file, target_file )

    def _sample_dataset( self, dataset_info, sample_properties, src_skip_set, trg_skip_set ):
        if not os.path.isfile( dataset_info.source_file ):
            raise Exception( f'Cannot sample file {dataset_info.source_file}, because it does not exists' )
        if not os.path.isfile( dataset_info.target_file ):
            raise Exception( f'Cannot sample file {dataset_info.target_file}, because it does not exists' )

        if sample_properties.size == 0:
            # We don't need to sample anything
            self.logger.info( f'Requested 0 lines for sample - we will not do anything' )
            return DatasetInfo( None, None, 0 ), dataset_info

        sample_prefix = 'sample'
        remain_prefix = 'remain'

        src_file_name = os.path.basename( dataset_info.source_file )
        trg_file_name = os.path.basename( dataset_info.target_file )

        src_file_dir = os.path.dirname( dataset_info.source_file )
        trg_file_dir = os.path.dirname( dataset_info.target_file )

        sample_src_file = os.path.join( src_file_dir, f'{sample_prefix}.{src_file_name}')
        sample_trg_file = os.path.join( trg_file_dir, f'{sample_prefix}.{trg_file_name}')

        remain_src_file = os.path.join( src_file_dir, f'{remain_prefix}.{src_file_name}')
        remain_trg_file = os.path.join( trg_file_dir, f'{remain_prefix}.{trg_file_name}')
        
        # Counting lines in both files to be sure that they at least the same length
        self.logger.info( 'Counting lines in source and target files' )
        file_1_count = buf_count_newlines_gen( dataset_info.source_file )
        file_2_count = buf_count_newlines_gen( dataset_info.target_file )
        if( file_1_count != file_2_count ):
            raise Exception( f'Source file {dataset_info.source_file} and target file {dataset_info.target_file} have different number of lines' )
        self.logger.info( f'Total lines: {file_1_count}' )
        self.logger.info( f'Requested sample lines: {sample_properties.size}' )

        random.seed( self.random_seed )
        if file_1_count <= sample_properties.size:
            sample_set = set( range(file_1_count) )
        else:
            sample_set = set( random.sample( range(file_1_count), sample_properties.size ) )

        # Let's keep track of lines that we already included into sample in case we don't want any duplicates
        src_sample_unique = set()
        trg_sample_unique = set()

        # Let's keep track of actual number of strings that we included in the sample
        sample_count = 0

        with open( dataset_info.source_file, 'r', encoding='utf8' ) as src_stream,\
            open( dataset_info.target_file, 'r', encoding='utf8' ) as trg_stream,\
            open( sample_src_file, 'w', encoding='utf8' ) as sample_src_stream,\
            open( sample_trg_file, 'w', encoding='utf8' ) as sample_trg_stream,\
            open( remain_src_file, 'w', encoding='utf8' ) as remain_src_stream,\
            open( remain_trg_file, 'w', encoding='utf8' ) as remain_trg_stream:
            for i, (src_line, trg_line) in tqdm( enumerate( zip( src_stream, trg_stream ) ), total= file_1_count, desc='Extracting sample' ):
                if i in sample_set \
                    and self._line_is_ok( src_line, sample_properties.min_str_length, src_skip_set, src_sample_unique ) \
                    and self._line_is_ok( trg_line, sample_properties.min_str_length, trg_skip_set, trg_sample_unique ):
                    sample_src_stream.write(src_line)
                    sample_trg_stream.write(trg_line)

                    sample_count += 1

                    if sample_properties.no_duplicates:
                        # If we don't want any duplicates in the sample, then let's keep already processed strings
                        src_sample_unique.add( src_line )
                        trg_sample_unique.add( trg_line )
                else:
                    remain_src_stream.write(src_line)
                    remain_trg_stream.write(trg_line)

        self.logger.info( f'Actual sample size after filtering is {sample_count}')

        return DatasetInfo( sample_src_file, sample_trg_file, sample_count ), DatasetInfo( remain_src_file, remain_trg_file )

    def _copy_dataset( self, dataset_info, dataset_name, source_language, target_language, sample_name ):
        if not dataset_info.source_file and not dataset_info.target_file:
            self.logger.info( f'No files were created for {sample_name} set')
            return DatasetInfo( None, None, 0 )

        if not os.path.isfile( dataset_info.source_file ):
            raise Exception( f'Cannot copy file {dataset_info.source_file}, because it does not exists' )
        if not os.path.isfile( dataset_info.target_file ):
            raise Exception( f'Cannot copy file {dataset_info.target_file}, because it does not exists' )

        result_folder = os.path.join( self.datasets_folder, f'{source_language}.{target_language}', dataset_name )

        if not os.path.isdir( result_folder ):
            os.makedirs( result_folder, exist_ok=False )

        src_final_file = os.path.join( result_folder, f'{sample_name}.{source_language}' )
        trg_final_name = os.path.join( result_folder, f'{sample_name}.{target_language}' )

        shutil.copy2( dataset_info.source_file, src_final_file )
        shutil.copy2( dataset_info.target_file, trg_final_name )
        
        return DatasetInfo( src_final_file, trg_final_name, dataset_info.size )

    @staticmethod
    def _create_strings_set( txt_file ):
        if not txt_file:
            return set()
        strings_set = set()
        with open( txt_file, 'r', encoding='utf8' ) as stream:
            for line in tqdm( stream, desc='Counting unique lines from previous samples' ):
                strings_set.add( line )
        return strings_set

    @staticmethod
    def _line_is_ok( line, min_length, skip_set, unique_set ):
        return len(line) >= min_length \
            and line not in skip_set\
            and line not in unique_set