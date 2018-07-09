# Download And Use WIKIPEDIA Sentences In TensorFlow

This is a repository using the Wiki Extractor to build and prepare WIKIPEDIA for use in tensorflow.

## Usage

First download the latest wikipedia xml dump, which is 16.7 GB in July 2018.

`ubuntu@ubuntu~:$ ./download_and_extract.sh`

Second build use the WikiExtractor to extract the raw text from the xml dump/

`ubuntu@ubuntu~:$ python ./wikiextractor/WikiExtractor.py --json ./enwiki-latest-pages-articles.xml`

Third, process the text into TensorFlow sequence examples.

`ubuntu@ubuntu~:$ prepare_for_tensorflow.sh`
