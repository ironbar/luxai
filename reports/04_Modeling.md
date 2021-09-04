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

**TODO:** I will go back to this point once I have created some hand-made agents and
I have a better understanding of the game.

## Iteration 1. Rule based agent

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

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
Below a picture of the viral agent with a big city before it collapses for lack of fuel.

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

## Iteration n. Iteration_title

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### n.1 Goal

### n.2 Development

### n.3 Results

### n.4 Next steps
