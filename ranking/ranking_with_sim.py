#tensorflow ranking with simulations

#load modules
from typing import Dict, Tuple
import tensorflow_ranking as tfr
from astropy import constants as const
from astropy import units as u
import csv
from matplotlib import pyplot as plt
import numpy as np
import os
from astropy.io import fits
import glob
from astropy.io import fits as pyfits
from astropy.table import Table, Column
#tensor packages
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from scipy.ndimage import zoom
import random

#import previous work
import ML_sim_trial as ML
import datasetforranking as dfr

#--------------------------------------------------------------------------------
#general parameters
filename_SL='./SL_sim_gal2/*.fits'
filename_random='./random_sim_gal/*.fits'
sample_size = 20
IMG_SIZE = 360

#--------------------------------------------------------------------------------
#create dataset for ranking
ranking_data, fitsNames = dfr.create_ranking_data(filename_SL, filename_random, sample_size)

# Convert the list to a NumPy array
ranking_array = np.array(ranking_data)

# Create a tf.data.Dataset from ranking_array
ratings = tf.data.Dataset.from_tensor_slices({
    'filename': tf.constant(ranking_array[:, 0, 0], dtype=tf.string),
    'sample_id': tf.constant(ranking_array[:, 0,1], dtype=tf.string),
    'rating': tf.constant(ranking_array[:, 0,2], dtype=tf.float32)
})

# Select the basic features.
ratings = ratings.map(lambda x: {
    "filename": x["filename"],
    "sample_id": x["sample_id"],
    "rating": x["rating"]
})

names_array = np.array(fitsNames)
# Create a tf.data.Dataset from fitsNames
filenames= tf.data.Dataset.from_tensor_slices({
    'filename': tf.constant(ranking_array[:], dtype=tf.string)
})#misschien onnodig

#select feature
samples = ratings.map(lambda x: x["sample_id"])
filenames = ratings.map(lambda x: x["filename"])

#--------------------------------------------------------------------------------
#build vocabularies

sample_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
sample_ids_vocabulary.adapt(samples.batch(1000))

filenames_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
filenames_vocabulary.adapt(filenames.batch(1000))

#--------------------------------------------------------------------------------
#group by sample_id to form lists for ranking models
key_func = lambda x: sample_ids_vocabulary(x["sample_id"])
reduce_func = lambda key, dataset: dataset.batch(100)
ds_train = ratings.group_by_window(
    key_func=key_func, reduce_func=reduce_func, window_size=100)

for x in ds_train.take(1):
  for key, value in x.items():
    print(f"Shape of {key}: {value.shape}")
    print(f"Example values of {key}: {value[:5].numpy()}")
    print()

#--------------------------------------------------------------------------------
#generate batched features and labels

@tf.autograph.experimental.do_not_convert #to repress warnings about autograph
def _features_and_labels(
    x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  labels = x.pop("rating")
  return x, labels

ds_train = ds_train.map(_features_and_labels)

ds_train = ds_train.apply(
    tf.data.experimental.dense_to_ragged_batch(batch_size=32))

for x, label in ds_train.take(1):
  for key, value in x.items():
    print(f"Shape of {key}: {value.shape}")
    print(f"Example values of {key}: {value[:3, :3].numpy()}")
    print()
  print(f"Shape of label: {label.shape}")
  print(f"Example values of label: {label[:3, :3].numpy()}")
  
  
#--------------------------------------------------------------------------------
#Define a model

class LensRankingModel(tf.keras.Model):

  def __init__(self, sample_vocab, filename_vocab):
    super().__init__()

    # Set up user and movie vocabulary and embedding.
    self.sample_vocab = sample_vocab
    self.filename_vocab = filename_vocab
    self.sample_embed = tf.keras.layers.Embedding(sample_vocab.vocabulary_size(),
                                                64)
    self.filename_embed = tf.keras.layers.Embedding(filename_vocab.vocabulary_size(),
                                                 64)

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    # Define how the ranking scores are computed: 
    # Take the dot-product of the user embeddings with the movie embeddings.

    sample_embeddings = self.sample_embed(self.sample_vocab(features["sample_id"]))
    filename_embeddings = self.filename_embed(
        self.filename_vocab(features["filename"]))

    return tf.reduce_sum(sample_embeddings * filename_embeddings, axis=2)

#--------------------------------------------------------------------------------
# Create the ranking model, trained with a ranking loss and evaluated with
# ranking metrics.
model = LensRankingModel(sample_ids_vocabulary, filenames_vocabulary)
optimizer = tf.keras.optimizers.Adagrad(0.5)
loss = tfr.keras.losses.get(
    loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
eval_metrics = [
    tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
    tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
]
model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)

#--------------------------------------------------------------------------------
#train and evaluate the model

model.fit(ds_train, epochs=3)

#generate predictions and value
# Get movie title candidate list.
for filenames in filenames.batch(2000):
  break

# Generate the input for user 4.
inputs = {
    "sample_id":
        tf.expand_dims(tf.repeat("4", repeats=filenames.shape[0]), axis=0),
    "filename":
        tf.expand_dims(filenames, axis=0)
}

# Get movie recommendations for user 4.
scores = model(inputs)
titles = tfr.utils.sort_by_scores(scores,
                                  [tf.expand_dims(filenames, axis=0)])[0]
print(f"Top 5 recommendations for user 4: {titles[0, :5]}")
  
  
  
  
  
  
  