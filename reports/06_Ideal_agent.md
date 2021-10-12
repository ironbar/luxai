# Ideal agent

On this document I will write how an ideal agent should be. This is a live document that might change
when new knowledge of luxai game is obtained.

## Wood for building cities, coal and uranium for fuel

| Resource | research points | Fuel value per unit | Units collected per turn | Cities can survive night after 20 turns of gathering | Turns need to build city |
|----------|-----------------|---------------------|--------------------------|------------------------------------------------------|--------------------------|
| wood     | 0               | 1                   | 20                       | 1.74                                                 | 5                        |
| coal     | 50              | 10                  | 5                        | 4.35                                                 | 20                       |
| uranium  | 200             | 40                  | 2                        | 6.96                                                 | 50                       |

A worker needs just 5 turns to gather enough wood for building a house on a single tree. On the
other hand it will take 50 turns to do the same with uranium. Clearly it has much more sense
to build with wood.

In the other hand a worker can only gather fuel for one city on 20 turns. In the same time
using coal will light 4 cities and uranium close to 7.

Then it seems to have sense to take wood and build houses around coal, or to build around forest
and carry coal and uranium to the forest. That opens the door to use carts because it will speedup
movement. Also building a "road" of houses that links forest to coal or uranium could have a lot
of sense.  
Probably building around the forest is better because minimizes movements. For each house we need
to travel less. Workers with uranium or coal will have to move just once a day or even less.  
Carts with a lot of resources may need bodygards because obstaculizing them will mean the collapse
of a city. Once we have a lot of workers some of them could devote to sabotage the other player.

## Multiple policies

At the start of the game we want to grow very fast to be able to gather coal and uranium while
preserving the forests.

Once we have access to coal or uranium the goal is to build the largest sustainable city possible.

Thus it seems to have sense to have multiple policies. That way we can train them with different
reward functions and this will simplify the training process.

## Competitive game

The behaviour of the agent should change if the enemy is near. If it is near we should be careful
to avoid losing resources. We could trap the enemy inside its forest just by standing on the surroundings
and prevent him from building houses.

**TODO:** play and save some captures here.

## A different policy for each era

### Wood Era

- Grow as fast as possible
- Advance to coal era as fast as possible
- Let the cities die at night
- Preserve the forest
- Block the enemy from entering our forests
- Expand to new areas as soon as possible
- At night each forest can only allocate as many workers as trees. Thus it does not have sense to have
more workers on a forest unless we want to protect from an invasion
- If there is an excess of workers send them to invade other forests
- When coal era is approaching send workers to conquer the coal resources

### Coal Era

- Now cities are built to be preserved, unless we use them as walls
- Coal areas are permanently settled and resources transferred to a city with carts
- When coal and forest are close is a good idea to surround coal with cities
- Advance to uranium era as fast as possible
- Send workers to conquer uranium when uranium era is approaching

### Uranium Era

- Be careful not to grow the cities so they are not sustainable
