[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache License 2.0][license-shield]][license-url]

# Machine Translation Evaluation framework

<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this repository is to provide useful framework to evaluate and compare different Machine Translation engines between each other on variety datasets.

The goal of evaluating Machine Translation quality have several complexities, like finding suitable test data, agreeing on the metric, attaching different translation engines and so on.

Our own goal was to build a system that allows us to compare several MT engines between each other on variety on datasets and to be able to repeat evaluation with some constant frequency. Surprisingly we cannot find any existing open source solution that will fit our requirements. So we created our own solution and decided to publish it in a case somebody will find it useful.

We're also providing current NMT evaluation results that we obtained during our own evaluation.

## Usage

First clone our repository, cd to the root folder and install all the required libraries
```bash
pip install -r requirements.txt
```

Evaluation happens in 3 stages:
- Download and prepare test dataset
- Translate all datasets with required MT engines
- Evaluate translations by required metrics

All evaluations scripts are stored in evaluation folder.

### To reproduce our results

```bash
# Scripts are called as modules (python -m ...) rather then scripts
# to simplify import for siblings folders
python -m evaluation.01_download_datasets
python -m evaluation.02_translate_datasets
python -m evaluation.03_evaluate_translations
# Final evaluation results will be in benchmarks folder:
# benchmarks/main_evaluation_set.tsv
# benchmarks/main_evaluation_set.tsv

# Careful – it took about a day to run the whole evaluation, even on a good GPU machine
# Careful – if you are providing your own keys for cloud services it will cost you money (30$ for Azure and 50$ for Google for this evaluation).

```

These three commands will download the same datasets that we use, and evaluate them against the same engines that we evaluated

### If you want to run your own evaluation

#### If your dataset can be downloaded from OPUS

1. Create your own yaml config in configs/dataset_to_download
   
   Config file should contain source and target language and list of datasets that we want to download from OPUS

```yaml
source_language: en
target_language: es

datasets:
    # Name of the dataset should be one from
    # https://opus.nlpl.eu/opusapi/?corpora=True
  - name: TED2013 
    # Type is your own tag to differntiate between datasets
    type: General
    # test, dev, train are fixed subsets
    # You can skip any of them in the config but cannot add your own
    # We guarantee that there are no data duplicates between these three sets
    test:
      # Required number of lines in the dataset
      # Actual size may be smaller due to removal of duplicates and short lines
      size: 10
      # Do we allow duplicates in the dataset
      no_duplicates: true
      # Minimal length in characters of the string to be included in the dataset
      min_str_length: 30
    dev:
      size: 100
      no_duplicates: true
      min_str_length: 30
    train:
      size: 100
      no_duplicates: true
      min_str_length: 30
```

2. Run evaluation scripts specifying your config file

```bash
# Using quick_evaluation_set.yaml as an example
# configs for translation and evaluation steps will be created automatically
python -m evaluation.01_download_datasets datasets_to_download=quick_evaluation_set
python -m evaluation.02_translate_datasets datasets_to_translate=quick_evaluation_set
python -m evaluation.03_evaluate_translations datasets_to_evaluate=quick_evaluation_set
# Final evaluation results will be in benchmarks folder:
# benchmarks/quick_evaluation_set.tsv
# benchmarks/quick_evaluation_set.tsv
```

#### If your have your own dataset

You may want to evaluate translation quality on your own dataset or public data not from OPUS.

1. Create two parallel files from your dataset, one with source data and one with reference translated data.

2. Create your own yaml config in configs/dataset_to_translate
   
   Config file should contain source and target language and list of datasets that we want to translate.

```yaml
source_language: en
target_language: es

datasets:
  # For your own datasets you may use whatever name you want
- name: PrivateDataset
  # Type is your own tag to differntiate between datasets
  type: General
  # test, dev, train are fixed subsets
  # You can skip any of them in the config but cannot add your own
  test:
    # Path to source and reference file
    # They should be either absolute or relative to the project root folder
    source: NMT_datasets/en.es/PrivateDataset/test.en
    target: NMT_datasets/en.es/PrivateDataset/test.es
    # Number of lines in source and reference files
    size: 9
  dev:
    source: NMT_datasets/en.es/PrivateDataset/dev.en
    target: NMT_datasets/en.es/PrivateDataset/dev.es
    size: 87
  train:
    source: NMT_datasets/en.es/PrivateDataset/train.en
    target: NMT_datasets/en.es/PrivateDataset/train.es
    size: 90

```

2. Run evaluation scripts from the second script, specifying your config file

```bash
# Using my_private_dataset as an example
# config for evaluation step will be created automatically
python -m evaluation.02_translate_datasets datasets_to_translate=my_private_dataset
python -m evaluation.03_evaluate_translations datasets_to_evaluate=my_private_dataset
# Final evaluation results will be in benchmarks folder:
# benchmarks/my_private_dataset.tsv
# benchmarks/my_private_dataset.tsv
```

#### If your have your own machine translation engine

If your MT engine is callable from Python code you can add it to this evaluation framework

1. Create new class for your MT engine in translator folder.
   
   translator/translator_empty.py can be used as reference.

```Python
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

```

2. Add your engine to the enum in translator/create_translator.py
   
   You can set any class name for the engine, and value should be full path of your translator class

```Python
class Translator(Enum):
    Custom = 'translator.translator_custom.TranslatorCustom'
    ...
```

3. Add your engine to the configs/translation_engines/engines.yaml config

```yaml
  # Class name that you set on previous step
- class: Custom 
  # Any name that you want that will specify for you this MT engine
  # with current settings
  name: Custom.option.42
  # Type is your own tag to differntiate between engines
  type: CustomEngine
  settings:
    # Any settings that you need for the engine
    # they will be passed by name to your engine __init__ function
    option: 42
```

4. Run evaluation scripts as usual
   
   Or alternatively, on step 3, you can create your own engine config file and specify it while calling translation script

```bash
# Using my_private_dataset and my_private_engine as an example
# config for evaluation step will be created automatically
python -m evaluation.02_translate_datasets datasets_to_translate=my_private_dataset translation_engines=my_private_engine
python -m evaluation.03_evaluate_translations datasets_to_evaluate=my_private_dataset
# Final evaluation results will be in benchmarks folder:
# benchmarks/my_private_dataset.tsv
# benchmarks/my_private_dataset.tsv
```

## Evaluation setup

We evaluated several translation engines for English-to-Spanish translation.

All evaluation results presented here is valid on 7th of July 2022, with the libraries' version defined in requirements.txt and datasets and models downloaded on that day.

Please keep in mind that if you will try to reproduce our results on later date, they may change due to updated models, libraries and cloud MT engines.

### Evaluated MT engines
#### Cloud engines
- Azure MT engine
[https://azure.microsoft.com/en-us/services/cognitive-services/translator/](https://azure.microsoft.com/en-us/services/cognitive-services/translator/)

  For Azure MT engine to work you will need your own subscription for Azure MT service, and you will need to provide the key for this service in AZURE_MT_KEY environment variable.
- Google MT engine
[https://cloud.google.com/translate/](https://cloud.google.com/translate/)

  For Google MT engine to work you will need your own subscription for Google MT service. You will need to create service account for your service and download key.json file for it. And you will need to provide path to this file in GOOGLE_APPLICATION_CREDENTIALS environment variable.

  More information on using Google Cloud with service account can be found here: [https://cloud.google.com/translate/docs/setup#creating_service_accounts_and_keys](https://cloud.google.com/translate/docs/setup#creating_service_accounts_and_keys)
#### Pre-trained open source engines
- Marian NMT + Opus MT
  
  Marian MT is very efficient machine translation architecture. [https://marian-nmt.github.io/](https://marian-nmt.github.io/)

  Opus MT is a set of Marian models trained on public data from OPUS. [https://github.com/Helsinki-NLP/Opus-MT](https://github.com/Helsinki-NLP/Opus-MT)

  In our evaluation we used Opus models converted to PyTorch and included in transformers library. [https://huggingface.co/transformers/model_doc/marian.html](https://huggingface.co/transformers/model_doc/marian.html)
- NeMo Machine Translation
  
  NeMo is NVIDIA toolkit for conversational AI models. It includes a bunch of pre-trained models for different tasks including Machine Translation. [https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/machine_translation.html](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/machine_translation.html)

  For NeMo we evaluated two models, large (24 encoder layers and 6 decoder layers) and small (12 encoder layers and 2 decoder layers)
#### Multi-lingual engines

- M2M100 and MBart50 is two massive multi-lingual models from the Facebook Research. They support translation between 100 and 50 languages respectively. Usually declared strength of such models is ability to work on low-resource languages. But we included them in our evaluation to see how well they can perform on common English-Spanish translation pair.

  For both models we used their PyTorch Transformers versions.

  M2M100 description: [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125)

  We evaluated two M2M100 model versions, with 418M and 1.2B paramaters:
  - [https://huggingface.co/facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M)
  - [https://huggingface.co/facebook/m2m100_1.2B](https://huggingface.co/facebook/m2m100_1.2B)

  MBart50 description: [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401)

  We evaluated two MBart50 model versions, one truly multi-lingual with any possible transaltion direction and another trained to translate only from English:
  - [https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt)
  - [https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)

### Datasets
Our work will be impossible without the OPUS project: [https://opus.nlpl.eu/](https://opus.nlpl.eu/)

Opus provide collections of different public translation datasets with the API that allows to search and download datasets in one common format.

For our evaluation we used 6 datasets available at Opus. For each dataset we tried to create test set of approximately 5000 lines.

| Dataset | Test set size | Links | Description |
| --- | --- | --- | --- |
| EMEA | 4320 | [https://opus.nlpl.eu/EMEA.php](https://opus.nlpl.eu/EMEA.php), [http://www.emea.europa.eu/](http://www.emea.europa.eu/) | Parallel corpus made out of PDF documents from the European Medicines Agency |
| WikiMatrix | 5761 | [https://opus.nlpl.eu/WikiMatrix.php](https://opus.nlpl.eu/WikiMatrix.php), [https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix) | Parallel corpora from Wikimedia compiled by Facebook Research |
| TED2020 | 5249 | [https://opus.nlpl.eu/TED2020.php](https://opus.nlpl.eu/TED2020.php), [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813]) | A crawl of nearly 4000 TED and TED-X transcripts from July 2020 |
| OpenSubtitles | 3097 | [https://opus.nlpl.eu/OpenSubtitles-v2018.php](https://opus.nlpl.eu/OpenSubtitles-v2018.php), [http://www.opensubtitles.org/](http://www.opensubtitles.org/) | Collection of translated movie subtitles from opensubtitles.org |
| EUbookshop | 5313 | [https://opus.nlpl.eu/EUbookshop.php](https://opus.nlpl.eu/EUbookshop.php), [Parallel Data, Tools and Interfaces in OPUS](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf) | Corpus of documents from the EU bookshop |
| ParaCrawl | 5227 | [https://opus.nlpl.eu/EMEA.php](https://opus.nlpl.eu/EMEA.php), [http://paracrawl.eu/download.html](http://paracrawl.eu/download.html) | Parallel corpora from Web Crawls collected in the ParaCrawl project |
| CCAligned | 5402 | [https://opus.nlpl.eu/EMEA.php](https://opus.nlpl.eu/EMEA.php), [CCAligned: A Massive Collection of Cross-lingual Web-Document Pairs](https://www.aclweb.org/anthology/2020.emnlp-main.480) | Parallel corpora from Commoncrawl Snapshots |

### Metrics
While BLEU considered most common metrics in Machine Translation, there are other options available that may be more tuned for different use cases.

In our evaluation we implemented several metrics to be able to compare machine translation engines across different dimension.

In our use case, it turned out that all metrics generally agree with each other (if one engine was better than the other by one metric, it was better by all metrics). For that reason, we use only BLEU to show our final evaluation results.

But it is important to notice here that in other uses cases (specifically for fine-grained machine translation quality comparison) other metrics may prove to be more useful.

| Metric | Link | Description |
| --- | --- | --- |
| BLEU | [https://github.com/mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [https://www.aclweb.org/anthology/P02-1040.pdf](https://www.aclweb.org/anthology/P02-1040.pdf) |
| TER | [https://github.com/mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf](hhttps://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf) |
| CHRF | [https://github.com/mjpost/sacrebleu](https://github.com/mjpost/sacrebleu) | [https://aclanthology.org/W15-3049.pdf](https://aclanthology.org/W15-3049.pdf) |
| ROUGE | [https://github.com/pltrdy/rouge](https://github.com/pltrdy/rouge) | [https://aclanthology.org/W04-1013.pdf](https://aclanthology.org/W04-1013.pdf) |
| BERTScore | [https://github.com/Tiiiger/bert_score](https://github.com/mjpost/sacrebleu) | [https://arxiv.org/abs/1904.09675](https://arxiv.org/abs/1904.09675) |
| COMET | [https://github.com/Unbabel/COMET](https://github.com/mjpost/sacrebleu) | [https://aclanthology.org/2020.emnlp-main.213/](https://aclanthology.org/2020.emnlp-main.213/) |

## Evaluation results

All results can be found in benchmarks/main_evaluation_set.(tsv|yaml|xlsx)

### General comparison between engines

!["General comparison"](/benchmarks/images/GeneralComparison.png)

### Multi-lingual models

!["Multi-lingual models"](/benchmarks/images/MultilingualComparison.png)

### Beam search vs greedy

!["Beam search vs greedy"](/benchmarks/images/BeamvsGreedy.png)

### Metrics comparison

When we sort all MT engines by one metric (let's say BLEU) all other metrics also became sorted.

!["Metrics comparison"](/benchmarks/images/MetricsComparison.png)

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

<!-- MAINTAINERS -->
## Maintainers

- Anton Masalovich
  - GitHub: [TonyMas](https://github.com/TonyMas)
  - Email: anton.masalovich@optum.com

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This repo wouldn't be possible without:
- [langcodes](https://github.com/rspeer/langcodes)
- [sacrebleu](https://github.com/mjpost/sacrebleu)
- [rouge](https://github.com/pltrdy/rouge)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [COMET](https://github.com/Unbabel/COMET)
- [Hydra](https://hydra.cc/)
- [transformers](https://huggingface.co/docs/transformers/)
- [MarianMT](https://marian-nmt.github.io/)
- [NeMo](https://github.com/NVIDIA/NeMo/)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Optum/nmt.svg?style=for-the-badge
[contributors-url]: https://github.com/Optum/nmt/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Optum/nmt.svg?style=for-the-badge
[forks-url]: https://github.com/Optum/nmt/network/members
[stars-shield]: https://img.shields.io/github/stars/Optum/nmt.svg?style=for-the-badge
[stars-url]: https://github.com/Optum/nmt/stargazers
[issues-shield]: https://img.shields.io/github/issues/Optum/nmt.svg?style=for-the-badge
[issues-url]: https://github.com/Optum/nmt/issues
[license-shield]: https://img.shields.io/github/license/Optum/nmt.svg?style=for-the-badge
[license-url]: https://github.com/Optum/nmt/blob/master/LICENSE
