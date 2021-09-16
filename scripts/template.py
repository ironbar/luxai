import sys
import argparse


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)


def parse_args(args):
    epilog = """
    """
    description = """
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('config_path', help='Path to yaml file with training configuration')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
