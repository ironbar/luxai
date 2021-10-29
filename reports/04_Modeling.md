# Modeling

## Select modeling technique

<!---Document the actual modeling technique that is to be used. If multiple
techniques are applied, perform this task separately for each technique.
Many modeling techniques make specific assumptions about the data—for example,
that all attributes have uniform distributions, no missing values allowed,
class attribute must be symbolic, etc. Record any such assumptions made. --->

**TODO:** I will go back to this point once I have created some hand-made agents and
I have a better understanding of the game.

## Generate test design

<!---Describe the intended plan for training, testing, and evaluating the models.
A primary component of the plan is determining how to divide the available dataset
into training, test, and validation datasets.

Doing a plot of score vs train size could be helpful to decide the validation strategy

Depending on the size of the data we have to decide how we are going to use submissions.
The less the submissions the most confidence we can have on the score. However sometimes
the data distribution is very different, or the size of the data is small and we have
to make a lot of submissions. Sometimes is not easy to have a good correlation between
validation score and LB score
--->

I will start using a simple win ratio metric to verify that new agents are better
than the old ones. I should probably compute the win ratio against more than one
agent to verify that the new agent is robust.

## Iteration 1. Rule based agent

### 1.1 Goal

The goal is to identify the difficulties when making a rule based agent

### 1.2 Development

On a first step I'm going to start from the simple agent and make it better incrementally. All the work can be found on notebook `002_hand_made_agent`. I have refactorized the simple agent and later done the following updates:

1. Add research
2. Build new workers
3. Build new cities
4. Avoid collisions

With this I have created the first agent called `viral` that builds cities and workers as fast as possible.

#### 1.2.1 Ideas for improving the agent

It is surpring how complex behaviours arise from simple rules. We could improve by:

- Better movements. Learn to avoid obstacles, avoid going to a city if we want to build a new one
- Do not go over already used resources
- Careful at night, avoid stupid deaths
- Resources should be treated differently
- Using carts could have sense when moving to a new location, that way other units behind will move much faster
- It's interesting to see that a simple agent is able to consume all the resources in the map, or nearly all. but the match is pretty long.
- When using the random mode the unit can "forget" the task, goes to build a new home and suddenly decides to go to the closest city
- Sometimes there are bottlenecks where a lot of units are inside a house
- Moving to the closest resource may not be the best strategy if it is already crowded

We may have to switch the framework to a task based one. Maybe it's time to look at halite competition.

#### 1.2.2 Challenges found when creating an agent

- Agent dying in the night because it does not have resources to create light (seed 1)
- How to avoid collisions between units, how to coordinate the different goals of the units
- How to move to a new location avoiding obstacles

#### 1.2.3 Learnings

- I have realized that building new city tiles is much cheaper than surviving the night (100 resources vs 300 resources). That explains why agents in the forum do "only" worry about increasing the worker population and not about preserving houses.
- However I believe that maybe once coal or uranium are available it may have sense to preserve cities since using wood for cities is cheaper. Maybe carts could play a role there.
- Moving to center does not increase the cooldown
- Changing the board size while mantaining the seed creates a totally different board

### 1.3 Results

The agent `viral` scores around 974 on leaderboard, which is around position 100 on the leaderboard.
This agent beats the `simple_agent` around 94.5% of the times.

In this [link](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-22702173) visualizations of the agent can be seen.
Below a picture of the viral agent with a big city before it collapses for lack of fuel on this [episode](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-episode-26566016).

![viral_agent](res/viral_agent.png)

### 1.4 Next steps

I would like to think about game strategy. Given the resources of a map, what the ideal agent will do.
For example I could compute what a forest could sustain. I can also think of city planning because if
I build in all the surroundings of a resource then I cannot grow anymore. Thus it may have sense to leave
holes so the city can grow, but then I have to use guards to avoid the enemy entering inside.
Probably it may have sense to send settlers to new forest at the start of the game since the surroundings
of a resource are the easiest place to build cities. At day a worker can travel to any place that is
located at a manhattan distance smaller than the half of the remaining day steps. Blocking the way of the enemy
could also be an interesting behaviour.

In the other hand I believe I have to switch to a framework that assigns tasks to the agents. This
will solve the problem of "forgetting" the current task and I have the feeling it will allow to scale
more complex behaviours easily than the current initial approach. Moreover I have seen that the same
philosophy was used on [Halite challenge](https://recursive.cc/blog/halite-iii-postmortem.html)

I don't know yet if there is room for machine learning. Maybe in the part of choosing an strategy, looking
at the board and identifying the key resources, the ideal layout of the city... But probably execution
is better done by code.

## Iteration 2. Task manager bot

### 2.1 Goal

The goal is to implement a task manager bot. At the start of the turn the bot will assign different
tasks to the units and later it will coordinate the actions based on their priority.

Hopefully this approach will scale better to complex behaviour than the one used on the first bot.

### 2.2 Development

I started thinking about this approach when dealing with collisions and when balancing between building
cities or gathering resources. Later I found that a very similar approach was described on [Halite challenge](https://recursive.cc/blog/halite-iii-postmortem.html)
Thus seeing that it has sense I have decided to implement it. I would love to get a bot that is
able to beat the viral agent more than 90% of the times.

#### 2.2.1 Dwight

After a complete redesign of the agent improving how the agent moves, avoid going out at night and a few
more small changes I have created agent `dwight` that is able to beat `viral` 65% of the times.
However in the forum performs very similar. Also I have found that there is a public agent that
scores around 1250, much better than my ~980.

Some ways of improving:

- Better assignment of targets for agents. They all currently go to the same cell. To be able to do
that I have to provide the current tasks of the other agents.
- A method to choose between building a new city or gathering fuel seems to be very important
- Better handling of night. Planning is needed.
- Cities should be treated as obstacles when going to build a new one

It seems that planning is crucial. We need to look at the future to be able to maximize the number
of cities at the end of the game. Simplifications of the game could help to planning, for example
we could represent the board with a graph being the nodes the clusters of resources and the edges
the distances between clusters.

A policy may help planning, but each map is different so it seems difficult to have a single policy.
Instead planning offers a general solution for all the maps.

I could devote a lot of time thinking about good strategies, and it will be funny, but I think planning
could solve that by brute force and I will learn more by taking the planning approach. Also I could
take the agent from the forum to fight with.

The game is an optimization problem: how to maximize the cities built at the end of the game given
the map resources and the opponent behaviour. My intuition is that planning is the best solution
for that problem, there might be very good solutions with rule-based agents, very good learned
policies... but I believe planning is the best solution.

### 2.3 Results

After a refactorization of the agent I have created agent `dwight` that is able to beat `viral` 65% of the times.
However in Kaggle performs very similar or worse.

Meanwhile an agent called `working_title` has been released in the forum that is able to beat
my agents around 95% of the times.

### 2.4 Next steps

I believe I should focus on planning. I have a strong enough agent from the forum to fight with. On
previous hungry geese challenge I missed the oportunity to apply planning and now I have to try it.

I could try to optimize an agent for a single map. Instead of thinking which strategies work best
I might simply overfit an agent to a map to find what is the best strategy for that map. Visualizing
that match will likely show something very close to the perfect agent. With that knowledge I could
later try to generalize to all the maps.

## Iteration 3. Interface for playing the game

### 3.1 Goal

The goal of this iteration is to create an interface to be able to play the game and to explore
the game to find good strategies.

### 3.2 Development

The idea is to create a simple yet usable game interface. I will have to create a visual representation
of the game with all the information needed to take the decisions and an interface.

One option is to create the interface using opencv. I have experience with it an it may work. Another
could be to use jupyter widgets. The first is based on keyboard while the second one relies more on
mouse.

One interesting thing is that the game representation I'm going to build could be later used for
training a model. Thus I have to diferentiate between game representation and game rendering.

#### 3.2.1 OpenCV

While doing the first renderings on a notebook I have noticed that pyplot is slow on the notebook.
It takes 400ms to show an image while only 100ms to create it. Thus it seems that I should better
use opencv just for speed reasons. 
The idea is to first render the game and then add an overlay of movements and visualization of the
current unit.

I'm going to first read some tutorials because I remember there were problems with some opencv versions
for conda that were not able to show images, and also there were different types of windows.

On 21/09/2021 I have a working version of the game interface. Now It's time to start playing and write
how I play and what would be the optimal way of playing. I can use the checkpoints to play alternative
strategies.

#### 3.2.2 Learnings

- A cart is able to move during day as fast as a worker, but leaves a road level of 1.5 behind so
all units following it could move without stopping.
- It is possible to build road on resources like wood, uranium and coal
- A cart cannot gather resources, it can only carry them
- If a house is built on a road, then the cell loses its previous road level when demolished
- Thus using carts may have sense when we want to speedup movement (maybe to go to another resource, or
to move faster in the forest) and also to carry resources from a resource to the city.
- To mine coal or uranium I believe the best strategy is to have workers on mining positions without moving and transferring resources to a cart. This ensures that the enemy cannot take the resource and at the same time I can move the cargo to the nearest city.
- In the other hand I believe forest should be mined with workers because collection speed is much faster.
- Carts can really speed processes because once they have visited a place the workers can move twice as fast
- Cities can be walls that protect the forrest

### 3.3 Results

I have implemented a user interface for playing the game and played on two maps. Playing is slow
because the current base agent I'm using creates wrong actions and I have to delete them and create from
zero. Having a more human agent could speedup playing.

At this point I have a much better understanding of the game and I believe I know a better strategy than
the one that is being used by top agents on the leaderboard. I have to think of how I can create an agent
that can play with that strategy.

### 3.4 Next steps

I can think of two different paths from here, I have already chosen Imitation learning but let's
document the two for reference.

#### 3.4.1 Hard coded agent

I have already written rules of how the agent should play on [Ideal agent](06_Ideal_agent.md). It should
be possible to translate those ideas to code. I believe a test-drive approach could work very well
for this case. I will prepare pairs of game state and actions and verify that my agent works as expected.
The number of tests will grow as the implementation gets more complex. I can use the game interface
to create those pairs.

However I don't think I will learn too much from this approach. Moreover people has already been taking this
approach for a long time so it will be difficult to catch up. And it is not easy to write a hard coded
agent as I have already seen on previous iterations.

#### 3.4.2 Imitation learning

Given enough resources I'm sure RL will be able to find a better strategy. But I don't believe I have
those resources. After studying the game complexity and considering the slow simulation of the game 
I believe RL would be very difficult for this game.

In the other hand imitation learning might work. The biggest drawback of imitation learning is that 
the agent may act strangely when it reaches a state of the game that was not encountered before. 
My idea of overcoming that problem is to pretrain the agent on matches from the forum, 
and then fine-tune on human matches. That could be a good combination.

The beauty of this approach is that I don't have to hard-code the agent and I will learn more about
imitation learning. That is why I have chosen to follow this approach.

## Iteration 4. Imitation learning from leaderboard agents

### 4.1 Goal

The goal is to explore how good an agent can become just by imitation learning and how many matches
do we need to imitate an agent.

In the best scenario we will end this iteration having an agent that performs close to the best
people on the leaderboard.

### 4.2 Development

#### 4.2.1 Tasks

- [x] Download matches from the leaderboard
- [x] Create features from game state
- [x] Implement a conditioned Unet architecture
- [x] Training script
- [x] Agent that uses a model for playing, it will need some post-processing of the predictions
- [x] Optimize the model architecture (optimize validation loss, but also measure agent score to see if they are correlated)
- [ ] Optimize input features
- [ ] Optimize training strategy (single train, pretrain and fine-tuning)

#### 4.2.2 Download matches from the leaderboard

As a start point I have a [notebook from the forum](https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader)
and also for hungry geese I developed my own [notebook on colab](https://colab.research.google.com/drive/1SBm3BG0ZvlDKuiu02zuLrZ7oU6Dsp-se).

I remember that it was only possible to download 10k matches each session, that is why I started
using colab to avoid that limitation.

I only want to download matches from the best agents. To do so I should rank the agents based on their
latest score.

I have created a [kaggle notebook](https://www.kaggle.com/ironbar/select-agents-for-downloading-matches) to
select matches for downloading them. Now I'm going to download the matches using [colab](https://colab.research.google.com/drive/1XtHHPVzrSnLGoqZ_A0CKdz21gSFkN_CI?usp=sharing).

#### 4.2.3 Create features from game state

Board cell features:

- [x] Resources
- [x] Road level

Board global features:

- [x] Day/night cycle
- [x] Turn
- [x] Is Last day?

Player cell features:

- [x] Units
- [x] Cities
- [x] Cooldown
- [ ] City size (I don't have this feature available, but I don't think is very important)
- [x] Cargo
- [ ] is active (I'm not adding this one because I have set negative values for cooldown for an active city)
- [x] Fuel that can be gathered a turn (take into account research points)
- [x] Resources that can be gathered a turn (take into account research points)
- [x] Number of turns a city can survive at night

Player global features:

- [x] Research points (normalized to coal and uranium era)
- [x] Number of cities
- [x] Number of units

This [Imitation learning notebook](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning)
has a very interesting way to compute the input features because it does not use the object provided
by luxai but directly parses the observation.

I need to compute the features, the actions and the masks for the predictions.

#### 4.2.4 Implement a conditioned Unet architecture

I want the model to generate actions for all the units and buildings at the same time. I also want to
use global information for taking the actions. Thus Unet seems like a very good fit for this problem.
However I also need to provide information such as day/night cycle, research points... So I have searched
for a conditioned Unet and I have found this paper [Conditioned-U-Net: Introducing a Control Mechanism in the U-Net for Multiple Source Separations](https://arxiv.org/abs/1907.01277)

I have read the paper and it seems the right way to tackle the problem. The ouput of the model will 
have binary cross-entropy loss because sometimes there are multiple actions for the same cell (imagine a unit inside a house)

It already has a [github implementation](https://github.com/gabolsgabs/cunet) that uses keras, so I
could try to use that code directly and see if it works.

Using that implementation as a start point I have prepared my own model that looks good. I also
need to implement a loss function that allows to use masks over the binary cross-entropy loss. Also
I have found that this implementation does not allow to use variable input size, so I will have to
add borders to boards smaller than 32x32.

##### 4.2.4.1 Tensorflow installation

I have found that in the kaggle notebooks tensorflow `2.4.0` is being used. However I had already installed
version `2.6.0` on my computer. To avoid losing time I will continue using `2.6.0` unless I find problems
when making the submissions. Thus I have an incentive to make a submission as soon as possible.

```bash
pip install tensorflow==2.6.0
pip install effortless_config
conda install -c conda-forge cudatoolkit==11.2.2 cudnn=8.1 -y
conda install -c conda-forge cudatoolkit-dev==11.2.2
pip install pydot graphviz
```

It is not founding `ptxas` even if I have installed `cudatoolkit-dev`.

#### 4.2.5 Training script

I have implemented a very simple training script. It loads 400 matches to memory because I cannot fit
more. On following iterations I will start using generators to overcome RAM limitations.

#### 4.2.6 Agent that uses a model for playing, it will need some post-processing of the predictions

I have implemented functions that allow to recover actions from predictions and verified that they
work when given the ground truth. Now I have to prepare an script that given a model path and
and output folder path it will create everything necessary to make a submission. I have to do this
as good as possible so I can reuse it later with new models.

I can take as a reference the work I did for hungry geese challenge.

```bash
cd "/mnt/hdd0/MEGA/AI/22 Kaggle/luxai/scripts/create_cunet_agent"
lux-ai-2021 --rankSystem="wins" --tournament  --maxConcurrentMatches 20 "clown/main.py" "working_title/main.py"
lux-ai-2021 "clown/main.py" "working_title/main.py"

```

```
Total Matches: 123 | Matches Queued: 39
Name                           | ID             | W     | T     | L     |   Points | Matches 
clown/main.py                  | RjsyUDGFuo9f   | 112   | 0     | 11    | 336      | 123     
working_title/main.py          | Amxi6tZ65x2Z   | 11    | 0     | 112   | 33       | 123     

#after fixing collisions
Total Matches: 521 | Matches Queued: 39
Name                           | ID             | W     | T     | L     |   Points | Matches 
working_title/main.py          | zjyVDttY1iGm   | 520   | 0     | 1     | 1560     | 521     
clown2/main.py                 | 7BhIkpq4HuJj   | 1     | 0     | 520   | 3        | 521     

```

I'm having trouble when uploading the model, it seems that the problem is that my luxai environment
has 3.9 python and kaggle uses 3.7. I was expecting this kind of problems and that's why I have
prepared this submission so soon. Thus I need to create a new conda environment and train the model again.

```bash
conda create -n luxai_37 pytest rope pylint tqdm numpy pandas scikit-learn ipython ipykernel coverage matplotlib python=3.7
source activate luxai
python -m ipykernel install --user --name $CONDA_DEFAULT_ENV --display-name "Python ($CONDA_DEFAULT_ENV)"
conda env export > environment.yml

conda create -n luxai_37 pytest rope pylint tqdm numpy pandas scikit-learn ipython ipykernel coverage matplotlib python=3.7
conda activate luxai_37
pip install tensorflow==2.6.0
pip install effortless_config
pip install pydot graphviz
conda install -c conda-forge cudatoolkit==11.2.2 cudnn=8.1 -y
pip install pyyaml
pip install kaggle-environments
pip install psutil
pip install nvidia-ml-py3
conda install -c conda-forge cudatoolkit-dev==11.2.2 -y
```

#### 4.2.7 Optimize the model architecture

I have run several experiments with the following conclusions:

- Data labels are very unbalanced so I have to give a weight of 32 to positive labels. Focal loss seems to help slighlty
- It seems that learning rate is not very relevant since I have tried with 2,4,8e-3 and results are almost the same.
- In the other hand increasing the batch size allows to reach training loss faster but maybe at the cost of poor generalization (not clear since val metrics are noisy)
- Unet filters: This has the greatest influence on train loss, the greater the number of filters the lower the loss.This suggest that the direction for improving is increasing the number of filters and increasing the training data
- Depth: 4 seems to be the optimum, 3 or 5 do not improve
- Condition: using a complex Film layer with a single hidden layer seems to be optimum

### 4.3 Results

I have been able to train agent `clown2` that achieves a LB score of 1358 and position 94/870. It is
far from the best agents (~1900) but it is a good start point for imitation learning.

### 4.4 Next steps

On the following iterations I want to scale up data. I will start by using data augmentation and I will
continue by using all the available data instead of using just the data that fits into memory.

## Iteration 5. Imitation learning with data augmentation

### 5.1 Goal

This is intended to be a short iteration where I implement data augmentation for training and measure
the improvement on the agent.

### 5.2 Development

Since the board is squared I can apply 90º rotations and also flips to multiply by 8 the amount of
training data.

I have to be careful with the actions of moving and transfer because the index of the layers should
be changed when applying the data augmentation. I will write tests to verify that data augmentation
is reversible.

I will also need to modify the training script to use a generator instead of an array of data.

#### 5.2.1 Pagliacci models

I have created a set of `pagliacci` models that have 32, 64 and 128 filters on the first layer of
the unet model. I have run two local tournaments:

```
Total Matches: 976 | Matches Queued: 39
Name                           | ID             | W     | T     | L     |   Points | Matches 
pagliacci_128/main.py          | RyKUtFiirJMv   | 305   | 7     | 107   | 922      | 419     
pagliacci_32/main.py           | bfgCCSIj0cc0   | 289   | 9     | 145   | 876      | 443     
pagliacci_64/main.py           | DbPktMtaVtIp   | 264   | 14    | 118   | 806      | 396     
clown2/main.py                 | oW1ycTIWUwaq   | 84    | 8     | 256   | 260      | 348     
working_title/main.py          | nygpdaZexM3E   | 15    | 0     | 331   | 45       | 346     


Total Matches: 898 | Matches Queued: 42
Name                           | ID             | Score=(μ - 3σ)  | Mu: μ, Sigma: σ    | Matches 
pagliacci_128/main.py          | uIEncBqBWeCr   | 26.0596907      | μ=28.554, σ=0.831  | 388     
pagliacci_64/main.py           | lwjqjgfaiHJ6   | 25.8941708      | μ=28.425, σ=0.844  | 397     
pagliacci_32/main.py           | Ev0DdTqxvQnq   | 23.9877369      | μ=26.396, σ=0.803  | 354     
clown2/main.py                 | TVRBKu8BvvLa   | 19.9573681      | μ=22.478, σ=0.840  | 347     
working_title/main.py          | 8FpBlwkv4lLC   | 13.2467371      | μ=16.224, σ=0.992  | 310     
```

I have realized that when using wins as the metric the number of matches is not asigned correctly. For
example notice that pagliacci_128 has played 20 matches less than plagiacci_32.

In the other hand using trueskill does not give interpretable results.

#### 5.2.2 Pretrain proof of concept

I would like to run a test to see if pretraining could improve generalization of the models. My idea
is to enable to load weights from a pretrained model and see if that improves the validation loss.
I will modify the csv file that is used for match selection to be able to do the experiment correctly.

We have pretrained with 800 matches and fine-tuned on 400 different matches.

| unet filters | pretrained best validation loss | from zero best validation loss |
|--------------|---------------------------------|--------------------------------|
| 32           | 0.2702                          | 0.2745                         |
| 64           | 0.2663                          | 0.269                          |
| 128          | 0.2599                          | 0.2621                         |

The validation loss with the pretrained model is consistently better but difference is not big.

#### 5.2.3 Training on a different number of matches

I have trained on 200, 400, 600, 800 and 1000 matches.

On this experiment I wanted to see the effect of using more matches for training. It is not obvious if that is good thing, because we are mixing
data from different agents, that play different. Probably is a good thing for pretraining but not sure about fine-tuning.

The validation set was different on each experiment, so that makes direct comparison impossible.

However we can see that the train losses relations are not trivial. They are not well organized and sorted from
less number of matches to biggest number of matches. This suggests that we have to do a deeper study in the
future.

Also I have to increase the limit of 50 epochs.

### 5.3 Results

I have created a set of `pagliacci` models that are currently scoring around 1450-1500 on leaderboard.
I'm currently on position 22/882 at day 28/10/2021.

### 5.4 Next steps

The next step is to implement a training script that is able to use all the data for training and
does not have RAM limitations. I will have to also experiment with the amount of data used for training
or fine-tuning the models.

## Iteration 6. Training on all the data

### 6.1 Goal

The goal of this iteration is to see if training with all the data yields improvements.

### 6.2 Development

#### 6.2.1 Training script

The idea is to load the matches in groups, for example of 50 and create batches from those matches
until there is no data left. Then load new matches and start again.

#### 6.2.2 Training on all the data

```bash
Total Matches: 177 | Matches Queued: 39
Name                           | ID             | W     | T     | L     |   Points | Matches 
pagliacci_32/main.py           | O1eks2TRoiMU   | 147   | 1     | 29    | 442      | 177     
napoleon_32/main.py            | 1jep9HhiTAJ2   | 29    | 1     | 147   | 88       | 177     
win/loss rate: 16/83%

Total Matches: 432 | Matches Queued: 40
Name                           | ID             | W     | T     | L     |   Points | Matches 
pagliacci_64/main.py           | ABc7rP1YXf9k   | 322   | 2     | 108   | 968      | 432     
napoleon_64/main.py            | sFcXldiaulXf   | 108   | 2     | 322   | 326      | 432     
win/loss rate: 25/74%

Total Matches: 136 | Matches Queued: 40
Name                           | ID             | W     | T     | L     |   Points | Matches 
pagliacci_128/main.py          | rKFXXW0q3g7q   | 62    | 29    | 45    | 215      | 136     
napoleon_128/main.py           | k71CkioerK2j   | 45    | 29    | 62    | 164      | 136     
win/loss rate: 33/45%

```

It seems that training in all the data creates weaker models. It also seems that the bigger the
model the difference it is able to learn better from all the data.

We need to see if this also holds in the public leaderboard. I have created a set of agents called `napoleon`
to test this.

### 6.3 Results

I have submitted a set of agents called `napoleon` and at the time of writing (29/10/2021) they score
1515, 1450 and 1344 for 128, 64, 32 filters respectively. Thus we can see the same pattern as the local
scores. Thus it seems that training in all the data yields worse results unless we use a very big model.

My hypothesis is that this happens because there are agents of very different skill, there is a big
difference between an agent scoring 1500 and another scoring 1800. Also the agents may have contradictory
policies that are harder to learn than simply learning from a single or a few agents.

### 6.4 Next steps

On the next iteration we are going to do the opposite. Instead of using data from all the agents we are going
to focus on training on a single or a few agents. Hopefully pretraining on all the data will be useful
on that little data trainings.

## Iteration 7. Focus on data from a singler or a few agents

### 7.1 Goal

Previous iteration has shown that training on all the data yields worse results if the model is small.
So now we are going to do the opposite, to train just on the best agent or on the best n agents.

### 7.2 Development

#### 7.2.1 Data exploration

Let's see how many matches do the best agents have.

| SubmissionId | n matches | FinalScore  |
|--------------|-----------|-------------|
| 23032370     | 176.0     | 1818.288755 |
| 22777661     | 394.0     | 1711.493722 |
| 23034455     | 244.0     | 1710.606181 |
| 22934122     | 241.0     | 1651.606058 |
| 22902113     | 345.0     | 1650.975344 |
| 23159502     | 65.0      | 1650.841907 |
| 23038485     | 137.0     | 1643.818123 |
| 23010508     | 195.0     | 1643.516025 |
| 23008895     | 191.0     | 1636.605564 |
| 22931784     | 233.0     | 1636.232316 |

This shows that `pagliacci` agents that were trained with 400 matches were a mixture of the best and the second best agent.

#### 7.2.2 Train on single agents

Let's see if training on matches from a single agent yields better results than previous models. I will
have to try both with and without pretraining.

| name                              | win rate | matches |
|-----------------------------------|----------|---------|
| focus_rank0_32_filters_pretrained | 63.8%    | 252     |
| pagliacci_32                      | 44%      | 270     |
| focus_rank0_32_filters            | 41.1%    | 260     |

Clearly the use of pretrained weights is beneficial. So I will always be using pretrained from now on. Let's now run a tournament between rank0, rank1 and rank2 agents.


#### 7.2.n Download more data from the cluster

I would also like to see if I can have information regarding the name of the team making the submission.

### 7.3 Results

The set of `focus` models

### 7.4 Next steps

Ensembling models, data augmentation at test, give different weight to the losses, it seems that
city action overfits first

## Iteration n. Iteration_title

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### n.1 Goal

### n.2 Development

### n.3 Results

### n.4 Next steps
