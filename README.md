# Table of Contents <img align="right" width="375" height="325" src="https://perks.optum.com/blog/wp-content/uploads/2016/08/olympics-doping.jpg"> 

- [Exploratory Notebooks](https://github.com/jwong853/Olympic_PED_use/tree/master/notebooks)
- [Report Notebook](https://github.com/jwong853/Olympic_PED_use/blob/master/reports/final_report/report.ipynb)
- [Project Presentation](https://github.com/jwong853/Olympic_PED_use/tree/master/reports/presentation)
- [Data](https://github.com/jwong853/Olympic_PED_use/tree/master/data)
- [src/ directory with project source code](https://github.com/jwong853/Olympic_PED_use/tree/master/src)
- [figures/ directory with project visuals](https://github.com/jwong853/Olympic_PED_use/tree/master/reports/figures)
- [Data references](https://github.com/jwong853/Olympic_PED_use/tree/master/references)
- [Project Conda environment](https://github.com/jwong853/Olympic_PED_use/blob/master/environment.yml)

# Context of Project
### Note on Project:
 - This project is currently on the results from the men's track events only. Olympedia's server went down while scraping the results from the rest of the events and for the women's events. The remaining results will be included in the next project iteration.

A predictive classification project for identifying PED use amongst Olympic Athletes during the Summer Olympic Games from 2012-2016. This project will be completed with binary classification modeling.

This project was completed with the intention of providing the International Olympic Committee (IOC) with an expedient tool aiding in the selection of athlete samples for re-testing. Doping in the world of Olympic Sports has been an issue for over 100 years. There have always been athletes willing to risk their health with the idea that performance enhancing drugs will help them win an event. Olympic organizations did not start testing athletes for PEDs until the late 60's. Since PED use had already been around, the scientists in charge with developing the tests to detect PEDs were at a disadvantage. The World Anti-Doping Agency (WADA) was established in 1999 to promote and coordinate the fight against doping in sports. WADA is an international independent agency which is equally composed and funded by sports organizations and governments of the world. 

The Anti-Doping Administration and Management System (ADAMS) was created in 2005, a database management system that simplifies the tasks of stakeholders and athletes invloved within the anti-doping system. ADAMs provides information such as athlete wherabouts, test planning and results, and lab results for WADA-accredited laboratories. Anti-doping agencies with WADA accrediation must follow the World Anti-Doping Code. This is a document harmonizing anti-doping policies, rules, and regulations within sport organizations. The code contains an International Standard for Testing and Investigations (ISTI) which covers establishment of athlete pool to be tested, prioritization between different sports and athletes, testing type prioritization, etc. Since the establishment of the World Anti-Doping Code, the detection of PED use has greatly increased but there are still athletes getting away steroid use.

I have scraped together multiple datasets containing olympic athletes from the Summer Games from 2012-2016 which holds the event results from Games for each athlete. I utilized this data to make a predictive classification model that uses event results for athletes as predictor variables.

The data for this project was gathered from many different sources and can be found below.

-[Athletes and statistics](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)

   - _Note: the data from the links below were scraped_

-[Athletes participating in track and field events](http://www.olympedia.org/athletes)

-[Event Results by year](https://www.olympic.org/olympic-results)

-[Event results](http://www.olympedia.org/games)


 - These result tables were downloaded in pdf format, opened using tabula-py, and saved as csv files. I am currently waiting for the server to come back up in order to get the rest of the results to use with the model.

-[Current elite athetes suspended or banned due to doping violations](https://www.athleticsintegrity.org/disciplinary-process/global-list-of-ineligible-persons?order-by=disqualificationPeriod&sort=desc&isDopingViolation=yes#filters)

-[Doping irrgularities at the Olympics](http://www.olympedia.org/lists/75/manual)


# Process

### EDA / Data Preprocessing

 - As discussed earlier the creation of this model was to aid the IOC in identifying PED use. Performing initial Exploratory Data Analysis allowed me to scrape together a dataset with only athletes that competed in the Summer Games from 2004-2016. To do this, I created individual tables for each event per year. These tables were then combined together and ordered by year of Olympic Games. 

 - Next, I repeated the above process for the results of each event. This wasn't as simple as some results show all heats, the semi-final, quarter-final, and final. While others only show the final result, not including atheletes from all of the individual heats. Do to this inconsistency I only gathered data that provided at least 2 race results per athlete. Since there are athletes that compete in more than one event as well as more than one Olympic Game, I chose to add these results to the dataset as individual columns. Since a majority of the datasets containing this information were from Olympedia, creating a dataframe and merging the results with the atheltes did not pose many issues.  

 - From there, I scraped multiple datasets containing athletes that had been flagged for PED use during the Olympics and only kept the data relevant for 2004-2016 Summer Sports. After cross-referencing the datasets, I ended up with a dataset containing 579 records for doping. To indicate whether or not an athlete had been found guilty of doping in the athelte dataset, I created a binary feature showing 1 for positive PED use and 0 for negative PED use. When trying to join this feature to the athlete dataframe, I was only successful in matching around 85 athletes from the doping datasets. Not all of the 579 athletes in the doping records were from the Summer Games, I found multiple athlete results from non-Olympic sporting events held during the same time period. After removing as many as I could find and formatting the names in the doping dataset, I was able to match on around 240 athletes. 
 
 - After combining these datasets, I needed to prepare the data for modeling. I began by filling the missing values for event results with 0 since the athlete had not participated in that event. Then, I subsetted the numerical features and categorical features seperately, Standard Scaling the numerical features and One Hot Encoding the categorical features. For the events such as the marathon where the event times were in hours, I converted to minutes. I also converted the event results provided in minutes, to seconds.

### Modeling
 - Since this was a binary classification problem, I chose to use a Dummy Classifier from Sci-Kit learn as my baseline model. I have decided to optimize the model for recall, limiting the amount of false negatives. I believe classifying a clean athlete as positive for doping won't have as much of an affect on the integrity of the sport. The athlete would more than likely be re-tested and able to prove that he/she was not doping. Classifying an athlete as clean when they aren't, allows athletes to get away with doping and possibly ruining the sport for clean athletes. 
 #### Models Used:
 - Dummy Classifier
 - Random Forest Classifier -On SMOTE resampled train and test
 - Decision Tree Classifier
 
 
 <h4 style="text-align:center;">FSM Dummy Classifier</h4>


|             | Precision | Recall | f1-score | Support | Avg Cross Val Score (3 splits, recall-scoring)       |                   
|-------------|-----------|--------|----------|---------|--------|
| **Negative_PED** |   98%    |  98%  |   98%   |  1031   |    
| **Positive_PED** |   0%    |  0%  |   89%   |  17   |  
|       |          |         |          |          |          5%                |

#### Evaluation

- This model was cross validated with the test split over 3 folds. I was not able to cross validate with the training data since I had very few positive doping cases. The results of this model were very poor, out of the 17 positive PED cases, the model did not successfully classify any of them as positive. 

#### Best Current Model

- The model that produced the best results at this point in the project was a Decision Tree Classifier. This classifier was fitted with the default parameters and then ran through a loop of different max_depths from 1-50. The greatest AUC was between the max_depths 5-10. Next I ran the model through a loop of different min_samples_split from 0.1-1.0. The greatest AUC here was between .1 and .15, the parameters I ended up using were:

 - max_depth: 7
 - min_samples_split: .13
 - criterion: entropy
 
 <h4 style="text-align:center;">Decision Tree Classifier</h4>


|             | Precision | Recall | f1-score | Support | Avg Cross Val Score (3 splits, recall-scoring)       |                   
|-------------|-----------|--------|----------|---------|--------|
| **Negative_PED** |   100%    |  100%  |   100%   |  1031   |    
| **Positive_PED** |   100%    |  82%  |   90%   |  17   |  
|       |          |         |          |          |          82%                |

#### Evaluation

- This model was cross validated with the test split over 3 folds. There was an increase in performance with the recall increasing to 82%. Out of the 17 positive PED cases, the model correctly flagged 14 of them.

- This model has the ability to classify positive PED use in athletes but I would like to improve on the class imbalance before deploying it to the International Olympic Committee.

### Next Steps

In order to improve on this project I would like to include the results from more sports. Cycling is one of the top three events where doping is most prominent. The event results for the cycling races provide statistics such as average speed, top speed, etc. I believe these results could benefit the model in classification since I could use the rate of change for the statistics provided.

I plan on implementing Deep Learning into this project if possible, specifically a Convolutional Neural Network.

### Project Reproduction

- [Project Conda environment](https://github.com/jwong853/Olympic_PED_use/blob/master/environment.yml)

If you would like to reproduce this project you can use the link provided above to create the same environment with the packages used here. If you have any issues recreating the environment, take a look over the documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). All of the datasets used can be recreated by running through the scraper cells in the notebooks. All models and test splits were done with a random state: 42. If you would like to see where I obtained all the data from, take a look in the [Data](https://github.com/jwong853/Olympic_PED_use/tree/master/data) folder of the repository. the EDA.ipynb notebook contains all the cells where the data is loaded and processed.

#### Contact Information

|         Name             |                  GitHub               | 
|--------------------------|----------------------------------|
|Jason Wong              |[jwong853](https://github.com/jwong853)|