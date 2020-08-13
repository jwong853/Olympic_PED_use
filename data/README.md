# Data

This directory contains all data used during the project.

| Folder/File | Description |
| ---- | ----------- |
| athletes_by_event | .csv files containing all athletes competing in track and field events during the Summer Olympic Games (2004-2016) |
| athletes_dataset | .zip and .csv files obtained from Kaggle Datasets containing Athletes in the Summer Olympics for the past 120 years |
| results_by_event | .csv files containing results from each track and field per Olympic Summer Games | 
| ADRVs_by_sport | .pdf files containing number of tests and doping positives (WADA has only release reports from years 2013-2017)
| model_df.csv | .csv file holding preprocessed dataframe to be used for modeling |
| model_df_2.csv | .csv file holding preprocessed dataframe with only men's events but also including years 2008-2016 |


## Links to other references
[Link to WADA/ADAMS](https://www.wada-ama.org)

[Link for event results](http://www.olympedia.org/games/results)

# Note on results_by_event folder:
I was scraping these event results when the website updated the tables, removing the results of athletes with positive doping
cases. Due to this, the results from this
folder are not currently being used. I intend
on going back and changing the format of the
csv's so they can be included in the future.