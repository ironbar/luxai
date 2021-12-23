# luxai

![luxai image](reports/res/project_picture.png)

Gather the most resources and survive the night!

https://www.kaggle.com/c/lux-ai-2021

## Methodology

I'm following [CRISP-DM 1.0](https://www.the-modeling-agency.com/crisp-dm.pdf) methodology for the reports.

I have skipped Evaluation and Deployment steps because they are not usually done on Kaggle.

1. [Business understanding](reports/01_Business_Understanding.md)
1. [Data understanding](reports/02_Data_Understanding.md)
1. [Data preparation](reports/03_Data_Preparation.md)
1. [Modeling](reports/04_Modeling.md)
1. [Solution summary](reports/05_Solution_Summary.md)
1. [Winning model documentation](reports/07_Winning_Model_Documentation.md)

* [Challenge workflow](reports/00_Challenge_Workflow.md)

## Code structure

     |_ luxai: library with code for the challenge
     |_ forum: all the scritps and notebooks taken from the forum
     |_ notebooks: jupyter notebooks made during the challenge. They start by number for easier sorting.
     |_ reports: documents made during the challenge according to CRISP-DM methodology
     |_ tests: folder with tests for the library
     |_ data: folder with light data from teh challenge
     |_ rules: the official rules of the challenge
     |_ agents: folder with agents for the challenge
     |_ scripts: scripts for training agents, playing the game...

## Hardware used during the challenge

My PC is described [here](https://pcpartpicker.com/b/jY8MnQ). It has two RTX3090 GPUs and the rest of the hardware can be seen below.

![ubuntu_specs](reports/res/ubuntu_specs.png)

## How to train a model

### 1. Select matches to download

I did this with this [kaggle public notebook](https://www.kaggle.com/ironbar/select-agents-for-downloading-matches/notebook). It simply ranks the submitted agents by score and saves a csv file with the agents exceeding a threshold score. 

2. Download the matches

3. Create train configuration file

4. Train

## How to create an agent
