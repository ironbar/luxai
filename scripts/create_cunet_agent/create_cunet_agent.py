"""
Creates an agent that uses a cunet model for playing
"""
import os
import sys
import argparse
import shutil


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    create_cunet_agent(args.model_paths, args.output_path,
                       args.horizontal_flip_augmentation, args.rotation_augmentation)


def create_cunet_agent(model_paths, output_path, horizontal_flip_augmentation, rotation_augmentation):
    os.makedirs(output_path)
    clone_cunet_repo(output_path)
    for idx, model_path in enumerate(model_paths):
        shutil.copyfile(model_path, os.path.join(output_path, 'model_%i.h5' % idx))
    download_luxai_sample_agent(output_path)
    create_cunet_agent_script(model_paths, output_path, horizontal_flip_augmentation, rotation_augmentation)
    shutil.copyfile('template_main.py', os.path.join(output_path, 'main.py'))


def clone_cunet_repo(output_path):
    os.system('git clone https://github.com/gabolsgabs/cunet.git %s' % output_path)
    os.system('rm -rf %s/*.txt %s/*.md %s/.*' % (output_path, output_path, output_path))


def download_luxai_sample_agent(output_path):
    cmd = 'wget https://github.com/Lux-AI-Challenge/Lux-Design-2021/raw/master/kits/python/simple/simple.tar.gz -O %s/simple.tar.gz' % output_path
    os.system(cmd)
    cmd = 'tar -C %s -xvf %s/simple.tar.gz' % (output_path, output_path)
    os.system(cmd)
    cmd = 'rm %s/simple.tar.gz %s/agent.py %s/main.py' % (output_path, output_path, output_path)
    os.system(cmd)


def create_cunet_agent_script(model_paths, output_path, horizontal_flip_augmentation, rotation_augmentation):
    module_paths = [
        '../../luxai/input_features.py',
        '../../luxai/output_features.py',
        '../../luxai/actions.py',
        '../../luxai/data_augmentation.py',
        'template_agent.py'
    ]
    text = ''
    for module_path in module_paths:
        with open(module_path, 'r') as f:
            text += f.read() + '\n'*3
    text = text.replace('kaggle_environments.envs.lux_ai_2021.test_agents.python.', '')
    text = text.replace('from luxai', '#from luxai')
    text = text.replace('__replace_original_model_path__', str(model_paths))
    text = text.replace('__replace_horizontal_flip_augmentation__', str(horizontal_flip_augmentation))
    text = text.replace('__replace_rotation_augmentation__', str(rotation_augmentation))
    with open(os.path.join(output_path, 'agent.py'), 'w') as f:
        f.write(text)


def parse_args(args):
    epilog = """
    python create_cunet_agent.py ../../agents/superfocus_64 /mnt/hdd0/Kaggle/luxai/models/19_ensemble/01_64_filters_rank0_pretrained_loss_weights_1_01_seed1/best_val_loss_model.h5

    python create_cunet_agent.py ../../agents/superfocus_64_ensemble /mnt/hdd0/Kaggle/luxai/models/19_ensemble/*/best_val_loss_model.h5
    """
    description = """
    Creates an agent that uses a cunet model for playing
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('output_path', help='Path to output folder where the agent will be created')
    parser.add_argument('model_paths', help='Path to .h5 cunet model file, it could be single model or multiple ones', nargs='*')
    parser.add_argument('--horizontal_flip_augmentation', default=1,
                        help='Number of augmentations with horizontal flip, set to 2 to enable it')
    parser.add_argument('--rotation_augmentation', default=1,
                        help='Number of augmentations with rotation, set to 4 for maximum augmentations')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
