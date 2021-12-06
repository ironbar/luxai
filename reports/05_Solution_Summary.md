# Solution Summary
<!---https://www.kaggle.com/wiki/WinningModelDocumentationTemplate --->

The final solution for the Lux AI challenge used Imitation Learning with a [Conditioned Unet](https://github.com/gabolsgabs/cunet). The conditioning mechanism was used to provide global information about the state of the game and
also the identify of the agent the model need to imitate. Thus the model learned to imitate multiple different
agents at the same time. By learning to imitate different agents the model generalized better, probably
because it had to learn better representations of the data. One funny thing of this approach is that
the model can imitate the different agents on prediction also.

## Training data

All matches from agents with a leaderboard score higher than 1700 on 01/12/2021 were used for training. That comprises a total of 82 different agents and close to 16k matches.

For validation 10% of the matches from the best leaderboard agent were used.

```bash
python create_multiagent_imitation_learning_training.py /mnt/hdd0/Kaggle/luxai/models/51_models_for_submissions/template.yml /mnt/hdd0/Kaggle/luxai/models/51_models_for_submissions 0 /home/gbarbadillo/luxai_ssd/agent_selection_20211201.csv 1700
```

## Features

All features are defined on `luxai/input_features.py`

### Map features

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

### Global features

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

## Model architecture

The model uses the [Conditioned Unet](https://github.com/gabolsgabs/cunet) with some variations to tune
the architecture for the challenge.

TODO: plot of the model, number of parameters

## Training

HW, training time, train params, data augmentation

## Agent

Copying the code, data augmentation

