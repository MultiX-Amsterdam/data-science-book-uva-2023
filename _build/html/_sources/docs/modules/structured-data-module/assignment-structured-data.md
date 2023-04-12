# Assignment (Structured Data Processing)

(Last updated: Feb 13, 2023)

## Materials

- [Structured Processing Tutorial Online Notebook](tutorial-structured-data)

## Usage

Follow the steps below to set up the assignment:
- Have the [JupyterLab](https://jupyter.org/install) environment ready.
- Download the structured data processing module from the [GitHub repository](https://github.com/MultiX-Amsterdam/structured-data-module). Or you can also download the zip file from [this link](https://github.com/MultiX-Amsterdam/structured-data-module/archive/refs/heads/main.zip).
- Open the notebook file and start doing the assignment.

Follow the steps below to do the assignment:
- Complete the assignments that are indicated in the tutorial notebook. You can leave out the optional assignment if it is too difficult or takes too much time.
- For each task, try your best to implement the solution without checking the answer.
- In the meantime, use the hint in the task description. You can also check the functions that are mentioned in the hint online.
- Try your best to implement the solution using the functions in the hint. If you still have no clue, check the answer, understand how it works, and implement the solution again by yourself.
- If there are parts that you do not understand, feel free to ask questions during the work sessions or on Canvas.

## Additional Recourses

Check the [pandas API](https://pandas.pydata.org/docs/reference/index.html) and [numpy API](https://numpy.org/doc/stable/reference/index.html) when writing the code.

## Optional Assignment

In the tutorial, you have learned the background of the Smell Pittsburgh application and the machine learning pipeline to build a model to predict the presence of bad smell based on sensor and weather data.
In this optional assignment, you need to design your own experiment to answer the following question raised by the local Pittsburgh community:

:::{admonition} Question of Community Concern
:class: note
What are the possible pollution sources that are related to the bad odor in the Pittsburgh region?
:::

### Learning Goals of the Optional Assignment

- Goal 1: Understand how to design experiments using machine learning pipelines to help stakeholders make sense of structured data.
- Goal 2: Be able to conduct experiments to critically reflect and understand how features and evaluation metrics affect the performance of machine learning models.
- Goal 3: Understand how to automate the machine learning pipeline of structured data to conduct experiments for different settings, document how the code works, and have good code quality.

### Tasks for the Optional Assignment

To answer this question, you need to select proper variables and fit the data to the model reasonably well.
Consider the following aspects when designing the experiment:
- How does the data look like?
  - Hint: Use the knowledge that you learned in this module to plot at least 1 graph for exploratory data analysis. The graph needs to be different from the ones that are already presented in the tutorial.
- What are the models that you want to use?
  - Hint: Use the knowledge that you learned in this module to choose a set of at least 3 different types of models that you want to investigate. Notice that this is a classification task, and a list of available models can be found at the [scikit-learn website](https://scikit-learn.org/stable/supervised_learning.html).
- What are the features that you are interested in exploring?
  - Hint: Use the knowledge that you learned in this module to select at least 3 different feature sets and check how these sets affect model performance. A list of available variables is mentioned in the tutorial and also can be found from the [dataset description](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction/tree/master/dataset/v2.1#description-of-the-air-quality-sensor-data).
  - Hint: Use the knowledge that you learned in this module to compute feature importance and inspect which are the important features.
- How much data does the model need to predict bad odor reasonably well?
  - Hint: Change the `train_size` parameter to increase or decrease the amount of data records for training the machine learning model. Try at least 3 different sizes.
- How often do you need to retrain the model using updated data?
  - Hint: Change the `test_size` parameter to indicate how often you want to retrain the model with updated data. Try at least 3 different sizes.
- How many hours do the model need to look back to check the previous data?
  - Hint: change the `b_hr_sensor` parameter in the `compute_feature_label` function to specify the number of hours that you want the model to look back. Try at least 3 different variations.
- It is recommended to use `f_hr_smell=7` in the `compute_feature_label` function (in Task 6 in the tutorial) and use value 40 as the threshold to indicate a smell event (in Task 7 in the tutorial). However, you are encouraged to think about this more and try different numbers. But keep in mind that these parameters affect the definition of a smell event, which means the rule that defines whether bad odors reach a certain level that requires community attention.

When designing the experiment, please consider the computation time carefully.
Keep in mind that if you have a very large set of features, training the machine learning model can take a very long time, and explaining the result can also be hard.
Instead of including all the available features in the experiment, it may be better to separate the features into several groups, and then train the machine learning model on different groups with different sets of features.

### Deliverable for the Optional Assignment

Your deliverable is **a Jupyter Notebook** that will be displayed online and sent to the Pittsburgh local community to help them understand air pollution patterns and advocate for policy changes.
Always keep the design brief mentioned above in mind when writing the deliverable.
In the deliverable, you need to explain how you completed the above-listed tasks and what the results are.

The deliverable also needs to contain all the code that you wrote, and **the code in your deliverable needs to be excecuted without errors**.
Having errors when running the code will significantly and negatively impact your score.

Specifically and importantly, **your deliverable MUST have the following sections**, where we will assess your learning outcome based on the grading rubric.
Failing to have these sections will significantly and negatively impact your score.
- **Summary**
  - Provide a summary of what you did and your findings. Maximum 150 words.
- **Experiment Design and Implementation**
  - Describe the machine learning models that you choose and explain why you choose them. The reasons may come from the knowledge that you learned in the lectures or the tutorial.
  - Describe the set of features (or multiple sets if you have multiple groups of features) that you choose and why you choose them. The reasons may come from the insights that you learned when you explored the data during the preparation phase of this module.
  -  Describe the number of hours that you want the model to look back, the amount of data that you use to train your model, and how often you think the model needs to be re-trained using updated data.
- **Experiment Results**
  - Produce and print one or multiple tables (using pandas dataframes) to show the results of your experiment. Make sure that you clearly describe what each column in the table means.
- **Discussion**
  - Explain your findings from the experiment. Keep in mind that the local community will read your findings, and they have limited knowledge of machine learning and data science. You need to explain the findings in a way that local citizens and policy-makers can understand.
  - For example, from the experiment results, which pollutants are important to predict the presence of poor odor? Are wind directions or speed important? You need to use the experiment result to support your argument, for example, the feature importance.

In the real world, we often need to provide convincing evidence to argue that the machine learning model fits the data reasonably well, for example, using the evaluation metrics mentioned in the tutorial.
The findings will not be convincing if the model fits the data poorly, like the dummy classifier (which always predicts “no” smell events) that we used in the tutorial.
Smell prediction is a hard task, so **do not worry too much about the low performance of the model** in this assignment.
The assignment aims to let you do experiments and compare the results, not optimize performance.

If you read articles (e.g., online blogs, academic papers) and take their ideas, you need to **cite and attribute the sources** in your deliverable. It is essential to keep this integrity in scientific research.

Generative AI is allowed in this assignment.
However, if you use generative AI to help you in completing this assignment, **you need to mention how you use the generative AI in detail**, including the type of tasks (e.g., proofreading text, brainstorming ideas, generating code), the model that you used (e.g., ChatGPT with GPT 3.5), the dates that you use the model, and the prompts that you enter to get the generative AI to create the content.
You also need to clearly indicate which parts in the deliverable are created with the support of generative AI.
Create an Appendix section in the deliverable to include the above-mentioned information regarding the usage of generative AI.

### Grading Rubric for the Optional Assignment

**Assessment of Learning Goal 1: Quality of describing the experiment design (45%)**
- Excellent (9-10)
  - The experiment design is rich and clearly justified about how such design can help identify local pollution patterns.
- Good (7-8)
  - The experiment design is adequate and properly justified about how such design can help identify local pollution patterns.
- Sufficient (6)
  - The experiment design is reasonable. But, the justification about how such design can help identify local pollution patterns may not be clear.
- Insufficient (<6)
  - The experiment design has low quality. The rationality and motivation of the experiment is poorly justified.

**Assessment of Learning Goal 2: Quality of critically reflecting the experiment and supporting the findings with evidence (45%)**
- Excellent (9-10)
  - The findings are rich, explained in a clear way that lay people can understand, and supported with strong evidence from the experiment results and visualizations.
- Good (7-8)
  - The findings are adequate, explained in a proper manner for lay people, and supported with proper evidence from the experiment results and visualizations.
- Sufficient (6)
  - Findings are provided. But some of them are not explained properly or not supported with evidence from the experiment.
- Insufficient (<6)
  - The findings have low quality, not explained well, and are not supported with evidence from the experiment.

**Assessment of Learning Goal 3: Ability to automate the experiment (weight 10%)**
- Excellent (9-10)
  - The experiment is fully automated (without errors when running the code), has good documentation about how the code works, and has very good code quality.
- Good (7-8)
  - The experiment is fully automated (without errors when running the code), has reasonable documentation about how the code works, and has the code be mostly human-readable.
- Sufficient (6)
  - The experiment is fully automated (without errors when running the code). There are some documentation about how the code works, but some may not be clear. Some parts of the code are hard to understand.
- Insufficient (<6)
  - The experiment is not automated. Or the code runs with errors. Or there are no documentation and poor code quality.
