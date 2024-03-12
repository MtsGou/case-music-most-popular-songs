# A STUDY WITH MOST POPULAR SONGS ACCORDING TO SPOTIFY FROM 1960 TO NOW

**Data science repository with files and codes used to analyze the most popular songs since 1960 -, investigate tendencies for the near future and further models to predict popularity (experimental).**

## Table of Contents

- [Context](#context)
- [Methods](#methods)
- [Conclusions](#conclusions)
- [Changes along the years and prediction for future](#changes-along-the-years-and-prediction-for-future)
  - [Song duration](#song-duration)
  - [Loudness](#loudness)
  - [Energy](#energy)
  - [Danceability](#danceability)
- [Other analyzes...](#other-analyzes)
  - [Time Signature](#time-signature)
  - [Minor vs Major?](#minor-vs-major)
  - [Most used Keys](#most-used-keys)
  - [Most Popular Genres](#most-popular-genres)
  - [Pattern among most popular (and also newer) songs](#pattern-among-most-popular-and-also-newer-songs)
- [Popularity prediction](#popularity-prediction)

----

## Context

According to a study published by Ohio State University, songs are becoming shorter, which would be the result of listeners’ attention spans shrinking in the streaming era. This can be realized when comparing songs introductions average duration in the 70's versus now. The study found that on average songs used to have introductions near to 20 seconds but now they are around five seconds long.

The study also discovered an increase in tempo of popular music. The premise is that "technological changes in the last 30 years have influenced the way we consume music, not only granting immediate access to a much larger collection of songs than ever before, but also allowing us to instantly skip songs."

So, this study now is to briefly analyze some tendencies and investigate not only the conclusions of Ohio's study, but of a vast collection of studies from all over the world that have converged in the same conclusion, to what concerns society behavior about consuming music.

----

## Methods

Data was collected from Kaggle's [dataset](https://www.kaggle.com/datasets/joebeachcapital/top-10000-spotify-songs-1960-now?resource=download) "Top 10000 spotify songs 1960 -now", whose data is directly extracted from [spotify API](https://developer.spotify.com/documentation/web-api/reference/get-audio-features).
It includes 10 000 of the most popular and relevant songs since the 60's according to billboard and ARIA.

Some packages and APIs were used: [Pyspark](https://spark.apache.org/docs/latest/api/python/index.html), [Spark ML](https://spark.apache.org/docs/latest/ml-guide.html), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Numpy](https://numpy.org/), [Scikit-Learn](https://scikit-learn.org/stable/), [TensorFlow](https://www.tensorflow.org/?hl=pt-br) and [Pydeequ](https://pydeequ.readthedocs.io/en/latest/README.html).

The study begins with an understanding section, statistical analysis and relevant queries to be made about the subject, followed by a cleaning and data preparation section for further models that predict average popular song duration for the next years until 2040. Also other variables prediction, such as loudness, energy and danceability. Finally, it was implemented some experimental models to try to predict the popularity of one song based on its characteristics, which proved to be challenging, as analysis show that this variable practically doesn't have a significant relation with the other variables.

-----

## Conclusions

In short, this study found that songs have become more danceable, more energetic, louder and shorter, and there is a tendency that this continues to go on in near future. But it is a difficult task to build a model that can precisely and efficiently predict a song's popularity from its characteristics such as duration, danceability, energy, acousticness, instrumentalness and so on.

----

## Changes along the years and prediction for future

### Song duration

![Average Duration vs Time](/plots/duration_time.png)

This shows that songs average lasting experienced a rapid growth from the 60's to the 80's, but since the 90's, average duration have been decreasing at a very steady and defined rate.

![Linear Decay Duration](/plots/corr_duration.png)

For the last 15 years, the average song duration has been decreasing linearly each year, more than 3 seconds each year.

![Predict average duration](/plots/pred_duration.png)

If song duration continues to decrease in the same rate, till 2040 songs could last, on average, near 2 minutes.

This model obtained scores of:

- RMSE : 3.48 [sec]
- MAE  : 2.82 [sec]
- R² score: 93.40 %

### Loudness

The range from 1990 to 2000 shows a very swift linear growth, but 2000 onwards the growth is softened, leading to a "halt" on approximately - 6.0 db in 2010 - 2020.

![Predict Average Loudness](/plots/pred_loudness.png)

So the prediction according to log model is that songs loudness continues to grow, but with a decreasing growing rate to the next years.

A logarithmic model was used and obtained scores of:

- RMSE : 0.42 [dB]
- MAE  : 0.37 [dB]
- R² score: 89.40 %

### Energy

Energy here means perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.

![Predict Average Energy](/plots/pred_energy.png)

According to this polynomial model, 'Energy' curve is still increasing, but coming from a increase to a 'halt' in the next years. This model obtained scores of 

- RMSE : 0.046 [range 0-1]
- MAE  : 0.035 [range 0-1]
- R² score: 60.68 %

### Danceability

![Predict Average Danceability](/plots/pred_danceability.png)

Danceability has a tendency to grow in the next years, following the pattern from 60's to 2020, and considering the 'jump' from 2016 to 2023. Obtained scores of

- RMSE : 0.033 [range 0-1]
- MAE  : 0.026 [range 0-1]
- R² score: 50.11 %

**[`🔼         back to top        `](#a-study-with-most-popular-songs-according-to-spotify-from-1960-to-now)**

----

## Other analyzes...

### Time Signature

It was found only three different musical time signatures among the most popular songs. As expected, the vast majority of songs (95,5%) use common time, 4/4 or 2/4, time signature. Only 3,91% use 3/4 time, and 0,58% use 5/4.

### Minor vs Major?

Most popular songs (69,8%) use major mode instead of minor.

| Mode | Count |
| -------- | -------- |
| Minor | 3014 |
| Major | 6977 |

### Most used Keys

![Keys](/plots/keys.png)

Found that the most used keys are C, G, A and D.

### Most popular genres

Most popular genres found, according to 'Popular' variable definition by Spotify, were K-Pop, Dance Pop, Hip Hop, Rap, Afrobeats, Art Pop and Electropop. It is important to mention that __'Popular' was defined by Spotify in a way it depends directly on how recent is the song.__

|  Artist Genres  |  Popularity  |
| -------- | -------- |
|    k-pop girl group|      97.0|
|big room,dance po...|      95.0|
|hip hop,rap,canad...|      95.0|
|afrobeats,nigeria...|      94.0|
|gen z singer-song...|      93.0|
|art pop,electropo...|      91.0|
|alt z,pop,pov: indie|      90.0|
|new romantic,new ...|      90.0|
|pop,uk pop,alt z,...|      90.0|
|electropop,pop,pe...|      89.0|

Among the most popular songs, 'danceability', 'energy', 'loudness' scored on average quite higher, and 'duration' scored almost one minute shorter comparing to least popular songs. 

### Pattern among most popular (and also newer) songs

Most Popular songs:

|danceability|energy| loud|duration|
| -------- | -------- | -------- | -------- |
|        0.68|  0.71|-5.77|198.67|


Least Popular songs:

|danceability|energy| loud|duration|
| -------- | -------- | -------- | -------- |
|        0.55|  0.58|-8.84|253.69|

This shows that the most popular songs tend to be more 'Danceable', 'Energetic', louder and shorter. But, as 'popularity' variable depends directly on how recent is the song, and the most popular songs here are also the newer in most cases, __this demonstrates also a tendency of newer popular songs to become more 'Danceable', 'Energetic', louder and shorter.__

**[`🔼         back to top        `](#a-study-with-most-popular-songs-according-to-spotify-from-1960-to-now)**

---- 

## Popularity prediction

* Feature variables: 'Track_duration', 'Danceability', 'Energy', 'Key', 'Loudness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo' and 'Year'.

* Target variable: 'Popularity'.

The mean error kept in the 25-30 range, considering that the variable is in range [0,100] that is quite an error. And the r2 score shows that the model couldn't ajust and therefore predict well and fit to data.

![Dataset correlation](/plots/corr_dataset.png)

This WAS expected, as popularity shows practically no correlation with other variables, and it doesn't appear to show any pattern.

![Dataset pairplot](/plots/pairplot.png)

Due to its random nature, it demonstrated to be a challenge to build a model that can predict popularity for songs from variables available on dataset. Although it is quite difficult to train machine learning models to predict this variable, this could be subject of further and deeper studies. And also, with the inclusion of 'Year' variable, the results were better.

Though the models found some of the most relevant features that impact on the results:

With random forest regressor model, the most important features found were __'Year', 'Track_duration', 'Loudness' and 'Danceability', in order of most to less relevant__.

With Lasso regression model, most important features according were 'Danceability', 'Valence', 'Instrumentalness' and 'Energy', but its results were significantly worse.

The best model found was the ANN model, but still can't fit to data, with an R² score of almost 10%.

**[`🔼         back to top        `](#a-study-with-most-popular-songs-according-to-spotify-from-1960-to-now)**

