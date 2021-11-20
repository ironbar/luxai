import sys
import argparse
import os
import yaml
import glob
import pandas as pd

INF = 100000

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)
    df = pd.read_csv(args.matches)
    args.score_thresholds.append(INF)

    for stage_idx, lower_threshold in enumerate(args.score_thresholds[:-1]):
        output_folder = os.path.join(args.output_folder, 'seed%i%s' % (args.seed, args.sufix), 'stage%i' % stage_idx)
        upper_threshold = args.score_thresholds[stage_idx + 1]
        sub_df = filter_dataframe_with_score(df, lower_threshold, upper_threshold)
        print('Stage %i matches: %i\tagents: %i' % (stage_idx, len(sub_df), len(sub_df.SubmissionId.unique())))
        train, val = get_train_and_val(sub_df, args.seed, divisions=max(10, int(len(sub_df)//100)))
        save_train_and_val_dataframes(train, val, output_folder)
        save_train_configuration(args.template, output_folder, stage_idx, len(train), len(val))
    print_commands_to_run_them_all(os.path.join(args.output_folder, 'seed%i%s' % (args.seed, args.sufix)))


def filter_dataframe_with_score(df, lower_threshold, upper_threshold):
    sub_df = df[df.FinalScore >= lower_threshold]
    sub_df = sub_df[sub_df.FinalScore < upper_threshold]
    sub_df.reset_index(drop=True, inplace=True)
    return sub_df


def save_train_and_val_dataframes(train, val, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    train.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_folder, 'val.csv'), index=False)


def get_train_and_val(df, seed, divisions=20):
    train = df.loc[[idx for idx in range(len(df)) if (idx + seed) % divisions]]
    val = df.loc[[idx for idx in range(len(df)) if not (idx + seed) % divisions]]
    return train, val


def save_train_configuration(template_path, output_folder, stage_idx, train_matches, val_matches):
    with open(template_path, 'r') as f:
        train_conf = yaml.safe_load(f)
    # update train and val dataframes
    for key in ['train', 'val']:
        train_conf['data'][key]['agent_selection_path'] = os.path.realpath(
            os.path.join(output_folder, '%s.csv' % key))
    # add pretrained weights if necessary
    if stage_idx:
        train_conf['model_params']['pretrained_weights'] = os.path.join(
            output_folder.replace('stage%i' % stage_idx, 'stage%i' % (stage_idx -1)),
            'best_val_loss_model.h5')
    # change the train and val steps
    train_conf['train_kwargs']['steps_per_epoch'] = min( 2000, int(train_matches*360/train_conf['data']['train']['batch_size']))
    train_conf['train_kwargs']['validation_steps'] = int(val_matches*360/train_conf['data']['val']['batch_size'])
    with open(os.path.join(output_folder, 'train_conf.yml'), 'w') as f:
        yaml.dump(train_conf, f, sort_keys=False)


def print_commands_to_run_them_all(output_folder):
    filepaths = sorted(glob.glob(os.path.join(output_folder, '*', 'train_conf.yml')))
    print()
    for filepath in filepaths:
        print('python train_with_generators.py %s' % filepath)
    print()


def parse_args(args):
    epilog = """
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
    parser.add_argument('score_thresholds', help='Score thresholds that will be used to create the training stages',
                        nargs='*', type=int)
    parser.add_argument('--sufix', help='Sufix to add to the name of the experiment', default='')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
