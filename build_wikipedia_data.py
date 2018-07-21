from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import glob
import os.path
import random
import sys
import threading



import nltk.tokenize
import numpy as np
from six.moves import xrange
import tensorflow as tf
import glove.utils


tf.flags.DEFINE_string("text_dir", "/tmp/train2014/",
                       "Wikipedia Text JSON directory.")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord file")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_integer("length_threshold", 64,
                        "Maximum allowed sentence length before cropping.")

FLAGS = tf.flags.FLAGS

SentenceMetadata = namedtuple("SentenceMetadata",
                           ["article_id", "article_title", "sentence_id", "sentence_words"])


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(sentence, vocab):
  """Builds a SequenceExample proto for a sentence..
  Args:
    sentence: A SentenceMetadata object.
    vocab: A Vocabulary object.
  Returns:
    A SequenceExample proto.
  """

  context = tf.train.Features(feature={
      "sentence/article_id": _int64_feature(sentence.article_id),
      "sencence/sentence_id": _int64_feature(sentence.sentence_id),
  })

  title_ids = [vocab.word_to_id(word) for word in sentence.article_title]
  sentence_ids = [vocab.word_to_id(word) for word in sentence.sentence_words]
  feature_lists = tf.train.FeatureLists(feature_list={
      #"sentence/article_title": _bytes_feature_list(sentence.article_title),
      "sentence/article_title_ids": _int64_feature_list(title_ids),
      #"sentence/sentence_words": _bytes_feature_list(sentence.sentence_words),
      "sentence/sentence_words_ids": _int64_feature_list(sentence_ids)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_sentences(thread_index, ranges, name, sentences, vocab,
                           num_shards):
  """Processes and saves a subset of sentences as TFRecord files in one thread.
  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    sentences: List of SentenceMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_sentences_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    sentences_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in sentences_in_shard:
      sentence = sentences[i]

      sequence_example = _to_sequence_example(sentence, vocab)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_sentences_in_thread))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote %d sentences to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d sentences to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, sentences, vocab, num_shards):
  """Processes a complete data set and saves it as a TFRecord.
  Args:
    name: Unique identifier specifying the dataset.
    sentences: List of SentenceMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Shuffle the ordering of sentences. Make the randomization repeatable.
  random.seed(12345)
  random.shuffle(sentences)

  # Break the sentences into num_threads batches. Batch i is defined as
  # sentences[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(sentences), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in xrange(len(ranges)):
    args = (thread_index, ranges, name, sentences, vocab, num_shards)
    t = threading.Thread(target=_process_sentences, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d sentences in data set '%s'." %
        (datetime.now(), len(sentences), name))


def _process_sentence(sentence):
  """Processes a sentence string into a list of tonenized words.
  Args:
    sentence: A string sentence.
  Returns:
    A list of strings; the tokenized caption.
  """
  tokenized_sentence = [FLAGS.start_word]
  tokenized_sentence.extend(nltk.tokenize.word_tokenize(sentence.strip().lower()))

  # Crop sentence if larger than threshold
  if len(tokenized_sentence) > (FLAGS.length_threshold - 2):
    tokenized_sentence = tokenized_sentence[:(FLAGS.length_threshold - 2)]
  tokenized_sentence.append(FLAGS.end_word)

  return tokenized_sentence


def _decode_file(filename_path):
  """Decodes a file into a list of Sentence Metadata objects."""
  with open(filename_path, "r") as f:
    all_pages = []
    num_articles = 0

    # Read all json objects in the file
    for x in f.readlines():
      num_articles += 1
      x = json.loads(x.strip())
      x_title = _process_sentence(x["title"])

      # Enumerate every sentence in the body text
      for i, s in enumerate(nltk.tokenize.sent_tokenize(x["text"])):

        # Build the metadata object
        x_meta = SentenceMetadata(
          article_id=int(x["id"]), 
          article_title=x_title, 
          sentence_id=i, 
          sentence_words=_process_sentence(s))
        all_pages.append(x_meta)

  return all_pages, num_articles


def _load_and_process_metadata(sentence_dir):
  """Loads sentence metadata from a JSON file.
  Args:
    sentence_dir: Directory containing directories of sentence files.
  Returns:
    A list of SentenceMetadata.
  """
  
  # Extract the filenames.
  sentence_filenames = glob.glob(os.path.join(sentence_dir, "*/*"))
  sentence_filenames += glob.glob(os.path.join(sentence_dir, "*"))

  # Process the sentences and combine the data into a list of SentenceMetadata.
  print("Processing sentences.")
  sentence_metadata = []
  num_articles = 0
  for base_filename in sentence_filenames:
    _meta, _len = _decode_file(base_filename)
    sentence_metadata.extend(_meta)
    num_articles += _len
  print("Finished processing %d sentences for %d articles in %d files" %
        (len(sentence_metadata), num_articles, len(sentence_filenames)))

  return sentence_metadata


def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Load sentence metadata from json files.
  wikipedia_train_dataset = _load_and_process_metadata(FLAGS.text_dir)

  # Create vocabulary from the training captions.
  train_words = [sentence.sentence_words for sentence in wikipedia_train_dataset]
  train_words += [sentence.article_title for sentence in wikipedia_train_dataset]
  vocab = glove.utils.load()[0]

  _process_dataset("train", wikipedia_train_dataset, vocab, FLAGS.train_shards)


if __name__ == "__main__":
  tf.app.run()




