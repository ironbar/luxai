import sys
import argparse
import os
import yaml
import glob
import pandas as pd

from create_curriculum_training import (
    filter_dataframe_with_score,
    save_train_and_val_dataframes
)

SUBMISSION_ID_TO_IDX_NAME = 'submission_id_to_idx.yml'


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)
    df = pd.read_csv(args.matches)
    df = filter_dataframe_with_score(df, args.score_threshold, int(1e6))

    n_agents = len(df['SubmissionId'].unique())
    print('A total of %i agents will be used' % n_agents)
    print(df.groupby('SubmissionId').head(1))

    output_folder = os.path.join(args.output_folder, 'seed%i%s' % (args.seed, args.sufix))
    os.makedirs(output_folder, exist_ok=True)
    create_submission_id_to_idx(output_folder, df)
    train, val = get_train_and_val(df, args.seed, divisions=args.folds)
    save_train_and_val_dataframes(train, val, output_folder)
    save_train_configuration(args.template, output_folder, len(train), len(val), n_agents, args.max_steps_per_epoch)
    print_commands_to_run_them_all(output_folder)


def create_submission_id_to_idx(output_folder, df):
    """
    Creates a yaml file that matches submission ids to indices for one hot encoding
    The indices are sorted from best to worse agent, so the index 0 is the best agent and -1 will be
    the worse
    """
    unique_submission_ids = df['SubmissionId'].unique()
    filepath = os.path.join(output_folder, SUBMISSION_ID_TO_IDX_NAME)
    with open(filepath, 'w') as f:
        yaml.dump({int(id): idx for idx, id in enumerate(unique_submission_ids)}, f, sort_keys=False)


def get_train_and_val(df, seed, divisions=20):
    val_indices = df[df['SubmissionId'] == df['SubmissionId'].unique()[0]].index
    val_indices = set(val_indices[seed::divisions])
    train = df.loc[[idx for idx in range(len(df)) if idx not in val_indices]]
    val = df.loc[[idx for idx in range(len(df)) if idx in val_indices]]
    print('\nTrain matches: %i\nVal matches: %i' % (len(train), len(val)))
    return train, val


def save_train_configuration(template_path, output_folder, train_matches, val_matches, n_agents, max_steps_per_epoch):
    with open(template_path, 'r') as f:
        train_conf = yaml.safe_load(f)
    # change model input features
    train_conf['model_params']['z_dim'] += n_agents
    # update train and val dataframes
    for key in ['train', 'val']:
        train_conf['data'][key]['agent_selection_path'] = os.path.realpath(
            os.path.join(output_folder, '%s.csv' % key))
        train_conf['data'][key]['submission_id_to_idx_path'] = os.path.realpath(
            os.path.join(output_folder, SUBMISSION_ID_TO_IDX_NAME))
    # change the train and val steps
    # I'm going to use half the steps to allow easier comparison between small and big trainings
    desired_steps_per_epoch = int(train_matches*360/train_conf['data']['train']['batch_size'])//2
    train_conf['train_kwargs']['steps_per_epoch'] = min(max_steps_per_epoch, desired_steps_per_epoch)
    print('desired_steps_per_epoch: %i (real %i)' % (desired_steps_per_epoch, train_conf['train_kwargs']['steps_per_epoch']))
    train_conf['train_kwargs']['validation_steps'] = int(val_matches*360/train_conf['data']['val']['batch_size'])
    with open(os.path.join(output_folder, 'train_conf.yml'), 'w') as f:
        yaml.dump(train_conf, f, sort_keys=False)


def print_commands_to_run_them_all(output_folder):
    filepaths = sorted(glob.glob(os.path.join(output_folder, 'train_conf.yml')))
    print()
    for filepath in filepaths:
        print('python train_with_generators.py %s' % filepath)
    print()


def parse_args(args):
    epilog = """
    python create_multiagent_imitation_learning_training.py /mnt/hdd0/Kaggle/luxai/models/49_submission_id_as_input/template.yml /mnt/hdd0/Kaggle/luxai/models/49_submission_id_as_input 0 /home/gbarbadillo/luxai_ssd/agent_selection_20211125.csv 1900 --sufix _agents3
    """
    description = """
    Creates the folders and files needed for curriculum training and also a script to do all
    the training stages
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('template', help='Path to yaml file with template training configuration')
    parser.add_argument('output_folder', help='Where to create the folders and files')
    parser.add_argument('seed', type=int, help='Seed for choosing training and validation matches')
    parser.add_argument('matches', help='Path to the csv file with all the matches')
    parser.add_argument('score_threshold', help='Score thresholds that will be used to create the training stages', type=float)
    parser.add_argument('--sufix', help='Sufix to add to the name of the experiment', default='')
    parser.add_argument('--folds', help='Number of partitions of the train data', default=10, type=int)
    parser.add_argument('--max_steps_per_epoch', help='Number of partitions of the train data', default=4000, type=int)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
