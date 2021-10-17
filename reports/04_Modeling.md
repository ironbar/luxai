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
- [ ] Implement a conditioned Unet architecture
- [ ] Training script
- [ ] Agent that uses a model for playing, it will need some post-processing of the predictions
- [ ] Optimize the model architecture (optimize validation loss, but also measure agent score to see if they are correlated)
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

##### 4.2.4.1 Tensorflow installation

I have found that in the kaggle notebooks tensorflow `2.4.0` is being used. However I had already installed
version `2.6.0` on my computer. To avoid losing time I will continue using `2.6.0` unless I find problems
when making the submissions. Thus I have an incentive to make a submission as soon as possible.

```bash
pip install tensorflow==2.6.0
pip install effortless_config
conda install -c conda-forge cudatoolkit==11.2
conda install -c conda-forge cudnn=8.1
conda install -c conda-forge cudatoolkit-dev==11.2.2
pip install pydot graphviz
```


### 4.3 Results

### 4.4 Next steps



## Iteration n. Iteration_title

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### n.1 Goal

### n.2 Development

### n.3 Results

### n.4 Next steps
