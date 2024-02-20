
#tutorial tensorflow_ranking

#load modules
from typing import Dict, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr

#elke user is een query. 
#In deze data proberen ze te voorspellen wat mensen vinden van films die ze nog 
#niet gezien hebben op basis van films die ze wel gezien hebben

#Voor ons is elke galaxy dus een movietitle, de rating is 1 of 0 en er is maar 1 user
#tenzij we crossvalidation gaan doen/gaan batchen


#%%capture --no-display
# Ratings data.
#read data, takes long to load
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

#%%capture --no-display
#Build vocabularies to convert all user ids and all movie titles into integer indices for embedding layers:
movies = movies.map(lambda x: x["movie_title"])
users = ratings.map(lambda x: x["user_id"])

user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
user_ids_vocabulary.adapt(users.batch(1000))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token=None)
movie_titles_vocabulary.adapt(movies.batch(1000))

#%%capture --no-display
#group by user_id to form lists for ranking models
key_func = lambda x: user_ids_vocabulary(x["user_id"])
reduce_func = lambda key, dataset: dataset.batch(100)
ds_train = ratings.group_by_window(
    key_func=key_func, reduce_func=reduce_func, window_size=100)

for x in ds_train.take(1):
  for key, value in x.items():
    print(f"Shape of {key}: {value.shape}")
    print(f"Example values of {key}: {value[:5].numpy()}")
    print()


#%%capture --no-display
#generate batched features and labels
def _features_and_labels(
    x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  labels = x.pop("user_rating")
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

#%%capture --no-display
#Define a model
class MovieLensRankingModel(tf.keras.Model):

  def __init__(self, user_vocab, movie_vocab):
    super().__init__()

    # Set up user and movie vocabulary and embedding.
    self.user_vocab = user_vocab
    self.movie_vocab = movie_vocab
    self.user_embed = tf.keras.layers.Embedding(user_vocab.vocabulary_size(),
                                                64)
    self.movie_embed = tf.keras.layers.Embedding(movie_vocab.vocabulary_size(),
                                                 64)

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    # Define how the ranking scores are computed: 
    # Take the dot-product of the user embeddings with the movie embeddings.

    user_embeddings = self.user_embed(self.user_vocab(features["user_id"]))
    movie_embeddings = self.movie_embed(
        self.movie_vocab(features["movie_title"]))

    return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)


#%%capture --no-display
# Create the ranking model, trained with a ranking loss and evaluated with
# ranking metrics.
model = MovieLensRankingModel(user_ids_vocabulary, movie_titles_vocabulary)
optimizer = tf.keras.optimizers.Adagrad(0.5)
loss = tfr.keras.losses.get(
    loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
eval_metrics = [
    tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
    tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
]
model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)


#%%capture --no-display
#train and evaluate the model

model.fit(ds_train, epochs=3)

#generate predictions and value
# Get movie title candidate list.
for movie_titles in movies.batch(2000):
  break

# Generate the input for user 42.
inputs = {
    "user_id":
        tf.expand_dims(tf.repeat("42", repeats=movie_titles.shape[0]), axis=0),
    "movie_title":
        tf.expand_dims(movie_titles, axis=0)
}

# Get movie recommendations for user 42.
scores = model(inputs)
titles = tfr.utils.sort_by_scores(scores,
                                  [tf.expand_dims(movie_titles, axis=0)])[0]
print(f"Top 5 recommendations for user 42: {titles[0, :5]}")

#%%capture --no-display
 






