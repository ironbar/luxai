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
    create_cunet_agent(args.model_path, args.output_path)


def create_cunet_agent(model_path, output_path):
    os.makedirs(output_path)
    clone_cunet_repo(output_path)
    shutil.copyfile(model_path, os.path.join(output_path, 'model.h5'))
    download_luxai_sample_agent(output_path)
    create_cunet_agent_script(model_path, output_path)
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


def create_cunet_agent_script(model_path, output_path):
    module_paths = [
        '../../luxai/input_features.py',
        '../../luxai/output_features.py',
        '../../luxai/actions.py',
        'template_agent.py'
    ]
    text = ''
    for module_path in module_paths:
        with open(module_path, 'r') as f:
            text += f.read() + '\n'*3
    text = text.replace('kaggle_environments.envs.lux_ai_2021.test_agents.python.', '')
    text = text.replace('from luxai', '#from luxai')
    text = text.replace('__replace_model_path__', os.path.realpath(os.path.join(output_path, 'model.h5')))
    text = text.replace('__replace_original_model_path__', model_path)
    with open(os.path.join(output_path, 'agent.py'), 'w') as f:
        f.write(text)


def parse_args(args):
    epilog = """
    python create_cunet_agent.py /mnt/hdd0/Kaggle/luxai/models/09_even_more_architecture_variations_around_condition/02_filters32_depth4_condition_8_complex/best_val_loss_model.h5 ../../agents/clown
    """
    description = """
    Creates an agent that uses a cunet model for playing
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('model_path', help='Path to .h5 cunet model file')
    parser.add_argument('output_path', help='Path to output folder where the agent will be created')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
