
# Include packages
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, TimestampType, BooleanType
from pyspark.sql.functions import col, year
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

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

# Path of data
path_data = "Music_case_study/top_10000_1960-now.csv"

def serialize(model):
    with open("rand_forest_model_popularity.pkl", "wb") as file:
        pickle.dump(model, file)

def extract_data():
    # Collect data spark df
    df = spark_session.read.csv(
        path_data,
        header=True,
        sep=',',
        schema=schema_music,
        timestampFormat="yyyy-MM-dd"
    )
    return df

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

def prepare_data_for_model(df):
    # Split train and test sets
    train, test = df.randomSplit([0.75, 0.25], seed = 42)

    # Input cols for the model
    input_cols = ['Track_duration', 
                'Danceability', 
                'Energy', 
                'Key', 
                'Loudness', 
                'Acousticness', 
                'Instrumentalness', 
                'Liveness', 
                'Valence', 
                'Tempo',
                'Year']

    assembler = VectorAssembler(
    inputCols = [x for x in train.columns if x in input_cols],
    outputCol = "features"
    )
    return assembler, train, test

def fit_model(assembler, train):
    # Random forest optimal params
    rf = RandomForestRegressor().setParams(
        numTrees=15,
        maxDepth=5,
        maxBins =32,
        labelCol = "Popularity",
        predictionCol = "prediction"
    )

    # Fit Random Forest Regression model
    model = Pipeline(stages = [assembler, rf]).fit(train)
    return model

def run():
    data = extract_data()
    data = clean_data(data)
    assembler, train, test = prepare_data_for_model(data)
    model = fit_model(assembler, train)
    serialize(model)

if __name__ == '__main__':
    run()
