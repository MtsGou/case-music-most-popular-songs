# Include packages
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType, BooleanType
from pyspark.sql.functions import col, year

# Start Spark Session
spark_session = (
    SparkSession.builder
    .appName("model_session")
    .getOrCreate()
)

# Define schema to read
schema_music = StructType([
    StructField('Track URI', StringType()),
    StructField('Track Name', StringType()),
    StructField('Artist URI', StringType()),
    StructField('Artist Name', StringType()),
    StructField('Album URI', StringType()),
    StructField('Album Name', StringType()),
    StructField('Album Artist URI', StringType()),
    StructField('Album Artist Name', StringType()),
    StructField('Album Release Date', TimestampType()),
    StructField('Album Image URL', StringType()),
    StructField('Disc Number', IntegerType()),
    StructField('Track Number', IntegerType()),
    StructField('Track Duration', IntegerType()),
    StructField('Track Preview URL', StringType()),
    StructField('Explicit', BooleanType()),
    StructField('Popularity', IntegerType()),
    StructField('ISRC', StringType()),
    StructField('Added By', StringType()),
    StructField('Added At', StringType()),
    StructField('Artist Genres', StringType()),
    StructField('Danceability', DoubleType()),
    StructField('Energy', DoubleType()),
    StructField('Key', IntegerType()),
    StructField('Loudness', DoubleType()),
    StructField('Mode', IntegerType()),
    StructField('Speechiness', DoubleType()),
    StructField('Acousticness', DoubleType()),
    StructField('Instrumentalness', DoubleType()),
    StructField('Liveness', DoubleType()),
    StructField('Valence', DoubleType()),
    StructField('Tempo', DoubleType()),
    StructField('Time Signature', IntegerType()),
    StructField('Album Genres', StringType()),
    StructField('Label', StringType()),
    StructField('Copyrights', StringType())
])

# Load model
def load_model():
    with open("rand_forest_model_popularity.pkl", "rb") as file:
        model = pickle.load(file)
    return model 

# Load data
def load_data(data = 'spotify_data.csv'):
    # Collect data spark df
    data = spark_session.read.csv(
        data,
        header=True,
        sep=',',
        schema=schema_music,
        timestampFormat="yyyy-MM-dd"
    )
    return data

def clean_data(df):
    # Dropping columns not used for model
    df = df.drop(
        'Track Name',
        'Artist Name',
        'Mode',
        'Explicit',
        'Track URI',
        'Artist URI',
        'Artist Genres',
        'Album URI',
        'Album Name',
        'Disc Number',
        'Track Number',
        'Album Artist URI',
        'Album Artist Name',
        'Album Image URL',
        'Speechiness',
        'Time Signature',
        'Track Preview URL',
        'ISRC',
        'Added By',
        'Added At',
        'Album Genres',
        'Label',
        'Copyrights',
    )

    # Changing 'Album Release Date' column to 'Year'.
    df = df.withColumn(
        'Year', year(col('Album Release Date'))
    ).drop(col('Album Release Date'))

    # Removing invalid for ['Energy', 'Valence', 'Track duration', 'Tempo', 'Loudness']
    df = df.filter(
        (col('Energy') <= 1) &
        (col('Valence') <= 1) &
        (col('Track duration') >= 10000.0) &
        (col('Tempo') >= 20.0) &
        (col('Loudness') <= 0.0)
    )
    return df

# Predict
def predict(data, model):
    return model.transform(data)

# Show predictions
def show_predictions(predictions):
    predictions.show()