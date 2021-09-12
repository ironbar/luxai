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

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

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

## Iteration n. Iteration_title

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

### n.1 Goal

### n.2 Development

### n.3 Results

### n.4 Next steps
