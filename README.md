# Table of Contents <img align="right" width="375" height="325" src="https://perks.optum.com/blog/wp-content/uploads/2016/08/olympics-doping.jpg"> 

- [Exploratory Notebooks](https://github.com/jwong853/Olympic_PED_use/tree/master/notebooks)
- [Report Notebook](https://github.com/jwong853/Olympic_PED_use/blob/master/reports/final_report/final_report.ipynb)
- [Project Presentation](https://github.com/jwong853/Olympic_PED_use/blob/master/reports/presentation/Olympic_ped_classification%20(2).pdf)
- [Data](https://github.com/jwong853/Olympic_PED_use/tree/master/data)
- [src/ directory with project source code](https://github.com/jwong853/Olympic_PED_use/tree/master/src)
- [figures/ directory with project visuals](https://github.com/jwong853/Olympic_PED_use/tree/master/reports/figures)
- [Data references](https://github.com/jwong853/Olympic_PED_use/tree/master/references)
- [Project Conda environment](https://github.com/jwong853/Olympic_PED_use/blob/master/environment.yml)

# Context of Project

A predictive classification project for identifying PED use amongst Olympic Athletes during the Summer Olympic Games from 2004-2016. This task will be completed with binary classification modeling.

My intention is to provide the International Olympic Committee (IOC) with an expedient tool aiding in the selection of athlete samples for re-testing. There are athletes willing to risk their health with the idea that performance enhancing drugs will help them win an event. Olympic organizations did not start testing athletes for PEDs until the late 60's *(Vlad, Hancu, Popescu, & Lungu, 2018)*.

  Since PED use had already been around, the scientists in charge with developing the tests to detect PEDs were at a disadvantage. The World Anti-Doping Agency (WADA) was established in 1999 to promote and coordinate the fight against doping in sports. WADA is an international independent agency which is equally composed and funded by sports organizations and governments of the world *(WADA, Who We Are 2020)*.


The Anti-Doping Administration and Management System (ADAMS) was created in 2005, a database management system that simplifies the tasks of stakeholders and athletes invloved within the anti-doping system. ADAMs provides information such as athlete wherabouts, test planning and results, and lab results for WADA-accredited laboratories *(WADA, ADAMS 2019)*. Anti-doping agencies with WADA accrediation must follow the World Anti-Doping Code. This is a document harmonizing anti-doping policies, rules, and regulations within sport organizations. The code contains an International Standard for Testing and Investigations (ISTI) which covers establishment of the athlete pool to be tested, prioritization between different sports and athletes, testing type prioritization, etc *(WADA, International Standard for Testing and Investigations (ISTI) 2020)*   *(WADA, ADAMS 2019)*.

I have scraped together multiple datasets containing olympic athletes from the Summer Games from 2012-2016 which holds the event results from Games for each athlete. I utilized this data to make a predictive classification model that uses event results for athletes as well as their weight, height, sex, and country they're representing as predictor variables.

# Data
- The data for this project was gathered from multiple sources and can be found below, the files are located in the [data directory](https://github.com/jwong853/Olympic_PED_use/tree/master/data).


|  folder/file | description | link  |
|------|---|---|
| athletes_dataset/athlete_events.csv  | Athletes that competed in the Summer Olympic Games from Athens-1896 to Rio-2016. Also includes data on weight, height, age, etc.  | [Kaggle](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)  |   |
|    athletes_by_event  | .csv files containing all athletes competing in track and field events during the Summer Olympic Games (2004-2016)   | [Olympedia](http://www.olympedia.org/athletes) |   |
| results_by_event | .csv files containing results from each track and field event per Olympic Summer Games  | [Olympedia](http://www.olympedia.org/games/results)  |   |
| results.csv  | results from each track and field event combined into a .csv  | [Olympic](https://www.olympic.org/olympic-results)  |   |
|  ADRVs_by_sport    | .pdf files containing number of tests and doping positives (WADA has only release reports from the years 2013-2017  | [Athletics Integrity](https://www.athleticsintegrity.org/disciplinary-process/global-list-of-ineligible-persons?order-by=disqualificationPeriod&sort=desc&isDopingViolation=yes#filters)  |     
| wiki_doping.csv  | List of doping cases in athletics | [Wikipedia](https://en.wikipedia.org/wiki/List_of_doping_cases_in_athletics)  |   |

## References 
- These references were used to gain insight to the data during analysis, they are located in the [references directory](https://github.com/jwong853/Olympic_PED_use/tree/master/references).

|  folder/file | description  | link  |
|---|---|---|
| wada-adrv-reports  | .pdf files containing testing figures and reports from 2014-2018  | [WADA](https://www.wada-ama.org/en/resources/laboratories/anti-doping-testing-figures-report)  |
| isti_2019_en_new.pdf  | .pdf file containing WADA International Standards for Testing and Investigation  | [WADA](https://www.wada-ama.org/en/resources/world-anti-doping-program/international-standard-for-testing-and-investigations-isti)  |
| world_anti-doping_code  | .pdf files containing anti-doping efforts through universal harmonization of core anti-doping elements  | [WADA](https://www.wada-ama.org/en/what-we-do/the-code?gclid=CjwKCAjwj975BRBUEiwA4whRB5XgXQ0d7geSLnC7Se_kJDpdU6_izKVa3HypFRkX0XljEV-dgUHVihoCIKUQAvD_BwE)  |

#### *Links to other References*

- These references were used to gain insight to the issues of doping in athletics.


|   |   |
|---|---|
| [Who We Are -WADA](https://www.wada-ama.org/en/who-we-are)  | *WADA, W. (2020, August 07). Who We Are. Retrieved August 16, 2020, from https://www.wada-ama.org/en/who-we-are*  |
| [Doping in Sports](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6311632/)  | *Vlad, R., Hancu, G., Popescu, G., &amp; Lungu, I. (2018, November). Doping in Sports, a Never-Ending Story? Retrieved August 16, 2020, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6311632/*  |
| [ADAMS](https://www.wada-ama.org/en/what-we-do/adams?gclid=CjwKCAjwj975BRBUEiwA4whRB8M8ZLGPAfpIiCYuysxY-_UO_yAwbv7VupBz6mg9sVjmOgyNTckjSRoCVPkQAvD_BwE)  | *WADA, W. (2019, August 01). ADAMS. Retrieved August 16, 2020, from https://www.wada-ama.org/en/what-we-do/adams?gclid=CjwKCAjwj975BRBUEiwA4whRB8M8ZLGPAfpIiCYuysxY-_UO_yAwbv7VupBz6mg9sVjmOgyNTckjSRoCVPkQAvD_BwE*  |
| [WADA Anti-Doping Code](https://www.wada-ama.org/en/what-we-do/the-code)  | *WADA, W. (2020, June 16). The Code. Retrieved August 16, 2020, from https://www.wada-ama.org/en/what-we-do/the-code*  |



# Process

### EDA / Data Preprocessing

 - As discussed earlier the creation of this model was to aid the IOC in identifying PED use. Performing initial Exploratory Data Analysis allowed me to scrape together a dataset with only athletes that competed in the Summer Games from 2004-2016. To do this, I created individual tables for each event per year. These tables were then combined together and ordered by year of Olympic Games. 

 - Next, I repeated the above process for the results of each event. Since there are athletes that compete in more than one event as well as more than one Olympic Game, I chose to add these results to the dataset as individual columns. Since a majority of the datasets containing this information were from Olympedia, creating a dataframe and merging the results with the atheltes did not pose many issues.  

 - From there, I scraped multiple datasets containing athletes that had been flagged for PED use during the Olympics and only kept the data relevant for 2004-2016 Summer Sports. After cross-referencing the datasets, I ended up with a dataset containing 579 records for doping. To indicate whether or not an athlete had been found guilty of doping in the athlete dataset, I created a binary feature showing 1 for positive PED use and 0 for negative PED use. When trying to join this feature to the athlete dataframe, I was only successful in matching around 85 athletes from the doping datasets. Not all of the 579 athletes in the doping records were from the Summer Games, I found multiple athlete results from non-Olympic sporting events held during the same time period. After removing as many as I could find and formatting the names in the doping dataset, I was able to match on around 240 athletes. 
 
 - After combining these datasets, I needed to prepare the data for modeling. I began by filling the missing values for event results with 0 since the athlete had not participated in that event. Then, I subsetted the numerical features and categorical features seperately, Standard Scaling the numerical features and One Hot Encoding the categorical features. 

### Modeling
 - Since this was a binary classification problem, I chose to use a Dummy Classifier from Sci-Kit learn as my baseline model. I have decided to optimize the model for recall, limiting the amount of false negatives. I believe classifying a clean athlete as positive for doping won't have as much of an affect on the integrity of the sport. The athlete would more than likely be re-tested and able to prove that he/she was not doping. Classifying an athlete as clean when they aren't, allows athletes to get away with doping and possibly ruin the sport for clean athletes. 
 #### Baseline and Best Models:
 - Dummy Classifier
 - Random Forest Classifier
 
 
 <h4 style="text-align:center;">FSM Dummy Classifier</h4>


|__Model__|__Training Accuracy__|__Testing Accuracy__|__Training Recall__|__Testing Recall__|__ROC-AUC score__|
|---|---|---|---|---|---|
|Dummy Classifier|83%|84%|9%|11%|52%|

#### Evaluation

- This model was cross validated with Stratified KFolds to preserve the distribution of positive and negative PED classes. 
- This model achieved a higher accuracy than I anticipated but I needed to improve upon the recall score for the purpose of identifying PED use. 
- An ROC-AUC score of 52% indicates this classifier is not performing well at distinguishing the positive and negative PED classes.

#### Best Model

- The model that provided the best results on the hold out testing data was a Random Forest Classifier with Random Undersampling of the majority class. 


 
 <h4 style="text-align:center;">Random Forest Classifier with Random Undersampling</h4>


|__Model__|__Training Accuracy__|__Testing Accuracy__|__Training Recall__|__Testing Recall__|__ROC-AUC score__|
|---|---|---|---|---|---|
|Random Forest w/ Undersampling| 88.72%  |83.33%   | 100%  | 75.41%  | 79.75%  |

#### Evaluation

- This model was cross validated with Stratified KFolds to preserve the distribution of positive and negative PED classes.

- The accuracy scores were not very far off from the testing and training but the recall score for the test is around 25% lower than the testing. This indicates the model is still overfitting on the training data.

- This model has the ability to classify positive PED use in athletes but I would like to improve on the class imbalance before deployment.
### Next Steps 

In order to improve on this project I would like to include the results from more sports. Cycling is one of the top three events where doping is most prominent *(WADA, Anti-Doping Testing Figures Report 2019)*. The event results for the cycling races provide statistics such as average speed, top speed, etc. I believe these results could benefit the model in classification since I could use the rate of change for the statistics provided.

I plan on implementing Deep Learning into this project if possible, specifically a Convolutional Neural Network.

 I would also like to spend more time during the initial combining of the datasets in order to match results on as many athletes as possible.

### Project Reproduction

- [Project Conda environment](https://github.com/jwong853/Olympic_PED_use/blob/master/environment.yml)

If you would like to reproduce this project you can use the link provided above to create the same environment with the packages used here. If you have any issues recreating the environment, take a look over the documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). All of the datasets used can be recreated by running through the scraper cells in the notebooks. All models and test splits were done with a random state: 42. If you would like to see where I obtained all the data, take a look in the [Data](https://github.com/jwong853/Olympic_PED_use/tree/master/data) folder of the repository. the EDA.ipynb notebook contains all the cells where the data is loaded and processed.

#### Contact Information

|         Name             |                  GitHub               | Email |
|--------------------------|----------------------------------|----------|
|Jason Wong              |[jwong853](https://github.com/jwong853)| jwong853@gmail.com |
