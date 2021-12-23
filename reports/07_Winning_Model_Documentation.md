# Winning model documentation

[Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines)

This is the Winning model documentation for LuxAI challenge from ironbar. All code and reports can be found at [github](https://github.com/ironbar/luxai) and also the latest version of this [document](https://github.com/ironbar/luxai/blob/main/reports/07_Winning_Model_Documentation.md).

## A. MODEL SUMMARY

<!---
General Guidelines
Keep in mind that this document may be read by people with technical and non-technical backgrounds and should aim to be informative to both.

Documentation should be in Word or PDF format. It should be in English (unless otherwise approved) and should be well-written.

The below should be considered helpful guidance. You can ignore any questions that are not relevant. You should also add useful details that are not covered by the questions.
 --->

### A1. Background on you/your team

- Competition Name: LuxAI
- Team Name: ironbar
- Private Leaderboard Score: 1791.8
- Private Leaderboard Place: 1791.8
- Name: Guillermo Barbadillo
- Location: Pamplona, SPAIN
- Email: guilllermobarbadillo@gmail.com

### A2. Background on you/your team

#### What is your academic/professional background?

My main interest is artificial intelligence. Since June 2014 I've been applying it at work and also on my free time on different projects and challenges. I have been lucky to win prizes in many international data science challenges.

I'm currently working on voice biometrics at Veridas.

#### Did you have any prior experience that helped you succeed in this competition?

I'm still learning about Reinforcement Learning but I have already participated on Hungry Geese challenge and also AnimalAI challenge.

#### What made you decide to enter this competition?

It was an oportunity to learn more about RL.

#### How much time did you spend on the competition?

Around 2 months.

### A3. Summary

<!---
4-6 sentences summarizing the most important aspects of your model and analysis, such as:

The training method(s) you used (Convolutional Neural Network, XGBoost)
The most important features
The tool(s) you used
How long it takes to train your model
--->

I have already summarized the solution on this [kaggle topic](https://www.kaggle.com/c/lux-ai-2021/discussion/293911).

The final solution for the Lux AI challenge used Imitation Learning with a [Conditioned Unet](https://github.com/gabolsgabs/cunet). The conditioning mechanism was used to provide global information about the state of the game and
also the identify of the agent the model need to imitate. Thus the model learned to imitate multiple different
agents at the same time. By learning to imitate different agents the model generalized better, probably
because it had to learn better representations of the data. One funny thing of this approach is that
the model can imitate the different agents on prediction also.

This approach worked because Toad Brigade agents were much better than the other teams. At the time of closing the submission period there is a difference of ~300 points in leaderboard between my agents and Toad Brigade's agents, it may go down because my best agents did not have time to converge but the difference is big. So the lesson learned is that using imitation learning
can provide a very strong agent, but 300 matches are not enough to get a good copy of the agent.

### A4. Features Selection / Engineering

#### What were the most important features?

<!---
We suggest you provide:
a variable importance plot (an example here about halfway down the page), showing the 10-20 most important features and
partial plots for the 3-5 most important features
If this is not possible, you should provide a list of the most important features.
--->

All features are defined on the repo at `luxai/input_features.py`

##### Map features

This involve features that have are 2d.

```python
CHANNELS_MAP = dict(
    wood=0, coal=1, uranium=2,
    player_worker=3, player_cart=4, player_city=5,
    opponent_worker=6, opponent_cart=7, opponent_city=8,
    cooldown=9, road_level=10,
    player_city_fuel=11, opponent_city_fuel=12,
    player_unit_cargo=13, opponent_unit_cargo=14,
    player_unit_fuel=15, opponent_unit_fuel=16,
    player_city_can_survive_next_night=17, opponent_city_can_survive_next_night=18,
    player_city_can_survive_until_end=19, opponent_city_can_survive_until_end=20,
    resources_available=21, fuel_available=22,
    player_is_unit_full=23, is_cell_emtpy=24, player_can_build_city=25,
    player_obstacles=26, playable_area=27,
)
```

I believe the names are self-explicative so please go to the code if you need more details for
each feature. All the features are normalized.

##### Global features

```python
FEATURES_MAP = dict(
    step=0, is_night=1, is_last_day=2,
    player_research_points=3, opponent_research_points=4,
    is_player_in_coal_era=5, is_player_in_uranium_era=6,
    is_opponent_in_coal_era=7, is_opponent_in_uranium_era=8,
    hour=9, city_diff=10,
    unit_free_slots=11,
)
```

In adittion to those features an extra feature is given representing the identity of the agent the
model needs to imitate. This is a simple one hot encoding. In the ohe the agents are sorted by leaderboard
score so first index is the best agent and last index is the worse. This can be used later on
prediction to choose which agent the model will imitate.

This features are feed to the conditioning branch.

#### How did you select features?

On a first step I implemented all the features that I thought were necessary. Then at the middle
of the challenge I run an [iteration](https://github.com/ironbar/luxai/blob/main/reports/04_Modeling.md#iteration-15-add-new-input-features) to find if adding new features could improve the models
and if removing some of them will also be beneficial.

#### Did you make any important feature transformations?

There is no magic feature, they all are pretty much trivial.

#### Did you find any interesting interactions between features?

Nothing relevant to say here

#### Did you use external data? (if permitted)

No

### A5. Training Method(s)

#### What training methods did you use?

#### Did you ensemble the models?

#### If you did ensemble, how did you weight the different models?

### A6. Interesting findings

#### What was the most important trick you used?

#### What do you think set you apart from others in the competition?

#### Did you find any interesting relationships in the data that don't fit in the sections above?

### A7. Simple Features and Methods

<!---
Many customers are happy to trade off model performance for simplicity. With this in mind:

Is there a subset of features that would get 90-95% of your final performance? Which features? *
What model that was most important? *
What would the simplified model score?

* Try and restrict your simple model to fewer than 10 features and one training method.

--->

### A8. Model Execution Time

<!---
Many customers care about how long the winning models take to train and generate predictions:
--->

#### How long does it take to train your model?

#### How long does it take to generate predictions using your model?

#### How long does it take to train the simplified model (referenced in section A6)?

#### How long does it take to generate predictions from the simplified model?

### A9. References

<!---
Citations to references, websites, blog posts, and external sources of information where appropriate.
--->

## B. SUBMISSION MODEL

<!---
Models should be submitted in a single zip archive that contains all of the items detailed below.

Below are some best practices for documenting and delivering your solution. There may be acceptable variations to these guidelines, depending on the type of competition, code, or methods you used. The core requirement is that you detail all the pieces needed by the host to reproduce your solution with the score your team achieved on the leaderboard within a reasonable margin.

This section is for a technical audience who are trying to run your solution. Please make sure your code is well commented.
--->

### B1. All code, data, and your trained model goes in a single archive

<!---
Except for data downloaded from Kaggle

Note: If you are preparing documentation for a Kernels Only competition, then please share your kernel with the host by adding their user name as a collaborator. If the winner's license requirement is open source, you can also make your kernel public.
--->

### B2. README.md

<!---
Create a README.md file at the top level of the archive. Here is an example file. This file concisely and precisely describes the following:

The hardware you used: CPU specs, number of CPU cores, memory, GPU specs, number of GPUs.
OS/platform you used, including version number.
Any necessary 3rd-party software, including version numbers, and installation steps. This can be provided as a Dockerfile instead of as a section in the readme.
How to train your model
How to make predictions on a new test set.
Important side effects of your code. For example, if your data processing code overwrites the original data.
Key assumptions made by your code. For example, if the outputs folder must be empty when starting a training run.
--->

### B3. Configuration files

<!---
Create a sub-folder with any necessary configuration files, such as `$HOME/.keras/keras.json`. The README should also include a description of what these files are and where they need to be placed to function.
--->

### B4. requirements.txt

<!---
Create a requirements.txt file at the top level of the archive. Here is an example file. This should specify the exact version of all of the packages used, such as `pandas==0.23.0`. This can be generated with tools like `pip freeze` in Python or `devtools::session_info()` in R. The requirements file can also be replaced with a Dockerfile, as long as the installations all use exact version numbers.
--->

### B5. directory_structure.txt

<!---
Create a readout of the directory tree at the top level of the archive. Here is an example file. This should be in the format generated by running the Linux command `find . -type d > directory_structure.txt` from the top level of your project folder.
--->

### B6. SETTINGS.json

<!---
This file specifies the path to the train, test, model, and output directories. Here is an example file.

This is the only place that specifies the path to these directories.
Any code that is doing I/O should use the appropriate base paths from SETTINGS.json
--->

### B7. Serialized copy of the trained model

<!---
Save a copy of the trained model to disk. This enables code to use the trained model to make predictions on new data points without re-training the model (which is typically much more time-intensive). If model checkpoint files were part of your normal workflow, the README should list the path to the folder you saved them in.
--->

### B8. entry_points.md

<!---
A list of the commands required to run your code. As a best practice, separate training code from prediction code. For example, if you’re using python, there would be up to three entry points to your code:

python prepare_data.py, which would
Read training data from RAW_DATA_DIR (specified in SETTINGS.json)
Run any preprocessing steps
Save the cleaned data to CLEAN_DATA_DIR (specified in SETTINGS.json)
python train.py, which would
Read training data from TRAIN_DATA_CLEAN_PATH (specified in SETTINGS.json)
Train your model. If checkpoint files are used, specify CHECKPOINT_DIR in SETTINGS.json.
Save your model to MODEL_DIR (specified in SETTINGS.json)
python predict.py, which would
Read test data from TEST_DATA_CLEAN_PATH (specified in SETTINGS.json)
Load your model from MODEL_DIR (specified in SETTINGS.json)
Use your model to make predictions on new samples
Save your predictions to SUBMISSION_DIR (specified in SETTINGS.json)
--->
