# Preparation (Structured Data Processing)

(Last updated: Feb 10, 2023)[^credit]

This part will prepare you with the background knowledge that we will use for this module.
Smell Pittsburgh is a mobile application for crowdsourcing reports of bad odors, such as those generated from air pollution.
The data is used to train machine learning models to predict the presence of bad smell, create push notifications to inform citizens about the bad smell, and explain local air pollution patterns.
You will need to read a paper and interact with the online visualization to explore data.

## Task 1: Read the Paper

First, read the following paper to get an idea about the motivation, background, and design of the Smell Pittsburgh application.
- Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao (Kenneth) Huang, and Illah Nourbakhsh. 2020. Smell Pittsburgh: Engaging Community Citizen Science for Air Quality. ACM Transactions on Interactive Intelligent Systems. 10, 4, Article 32. DOI:[https://doi.org/10.1145/3369397](https://doi.org/10.1145/3369397). Preprint:[https://arxiv.org/pdf/1912.11936.pdf](https://arxiv.org/pdf/1912.11936.pdf).

:::{warning}
You should already read this paper when preparing the second lecture.
If you come to the tutorial session without reading the paper, you will probably not be able to understand the data science pipeline well.
:::

When reading the paper, write down the answers to the following questions.
- Why is there a need to develop such an application in Pittsburgh?
- What are the data types that the Smell Pittsburgh application collects?
- How can the data be used potentially to help local people?
- What are the roles of data science in the Smell Pittsburgh project?

## Task 2: Explore Data

After you read the paper mentioned previously, explore the data in the following URL that visualizes smell reports and air quality data.
Please make sure you read the paper before doing this task.
- Link to Smell Pittsburgh data visualization: [https://smellpgh.org/visualization](https://smellpgh.org/visualization)

Specifically, please take a look at the following days to understand the distribution of data that have different conditions.
For example, some days look good, some days have very bad odors, and some days are in the middle of these two extremes.
- [Example of a bad smell day (Jul 7, 2020)](https://smellpgh.org/visualization?share=true&date=20200707&zoom=11&latLng=40.394,-79.914&city_id=1)
- [Example of a good smell day (Sep 23, 2021)](https://smellpgh.org/visualization?share=true&date=20210923&zoom=11&latLng=40.394,-79.914&city_id=1)
- [Example of a not really good smell day (Sep 20, 2021)](https://smellpgh.org/visualization?share=true&date=20210920&zoom=11&latLng=40.394,-79.914&city_id=1)

When investigating the patterns of bad odors, write down your answers to the following questions.
Smell events mean the occurance within a time range when many people complained about bad odors.
- Are there common wind patterns (indicated by the blue arrows on the user interface near the circles) when smell events are about to happen or are happening?
- Are there common patterns in air quality sensor measurements (indicated by the circle on the user interface with colors) when smell events are about to happen or are happening?
- Can you use the patterns that you found to identify similar smell events on other days? Find at least three other days that have similar patterns.

## Task 3: Check the Statistics

Next, after you explore the data, go to the following web page to see the aggregated statistics to understand the distribution of users and smell reports temporally and spatially.
- Link to the data analysis page: [https://smellpgh.org/analysis](https://smellpgh.org/analysis)

When checking the analysis on the above-mentioned web page, answer the following questions and write your answers down.
- Are there any characteristics about the distribution of smell reports over time and geographical regions?
- What are the common descriptions of bad odors that people reported?
- What are the possible predictors (e.g., chemical compounds, weather data) of bad smell in the Pittsburgh region?

[^credit]: Credit: this teaching material is created by [Yen-Chia Hsu](https://github.com/yenchiah).
