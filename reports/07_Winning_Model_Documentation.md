# Winning model documentation

[Winning Model Documentation Guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines)

## A. MODEL SUMMARY

<!---
General Guidelines
Keep in mind that this document may be read by people with technical and non-technical backgrounds and should aim to be informative to both.

Documentation should be in Word or PDF format. It should be in English (unless otherwise approved) and should be well-written.

The below should be considered helpful guidance. You can ignore any questions that are not relevant. You should also add useful details that are not covered by the questions.
 --->

### A1. Background on you/your team

- Competition Name:
- Team Name:
- Private Leaderboard Score:
- Private Leaderboard Place:
- Name: Guillermo Barbadillo
- Location: Pamplona, SPAIN
- Email: guilllermobarbadillo@gmail.com

### A2. Background on you/your team

#### What is your academic/professional background?

#### Did you have any prior experience that helped you succeed in this competition?

#### What made you decide to enter this competition?

#### How much time did you spend on the competition?

#### If part of a team, how did you decide to team up?

#### If you competed as part of a team, who did what?

### A3. Summary

<!---
4-6 sentences summarizing the most important aspects of your model and analysis, such as:

The training method(s) you used (Convolutional Neural Network, XGBoost)
The most important features
The tool(s) you used
How long it takes to train your model
--->

### A4. Features Selection / Engineering

#### What were the most important features?

<!---
We suggest you provide:
a variable importance plot (an example here about halfway down the page), showing the 10-20 most important features and
partial plots for the 3-5 most important features
If this is not possible, you should provide a list of the most important features.
--->

#### How did you select features?

#### Did you make any important feature transformations?

#### Did you find any interesting interactions between features?

#### Did you use external data? (if permitted)

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
