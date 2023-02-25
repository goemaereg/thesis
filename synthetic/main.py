import argparse
from dataset import prepare_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=None)

    parser.add_argument('--mode', default='prepare_data', type=str, help=None)
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary

    if args.mode == 'prepare_data':
        parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, help=None)
        parser.add_argument('--project', default='proj4', type=str, help=None)
        parser.add_argument('--n_classes', default=10, type=int, help=None)
        parser.add_argument('--n_samples', default=1000, type=int, help=None)
        parser.add_argument('--data_mode', default='train', type=str, help=None)
        parser.add_argument('--n_instances', default=1, type=int, help=None)
        parser.add_argument('--random_n_instances', action=argparse.BooleanOptionalAction, default=False, help=None)
        parser.add_argument('--type_noise', action=argparse.BooleanOptionalAction, default=False, help=None)
        parser.add_argument('--overlapping', action=argparse.BooleanOptionalAction, default=False, help=None)
        parser.add_argument('--background', action=argparse.BooleanOptionalAction, default=False, help=None)

        args, unknown = parser.parse_known_args()
        dargs = vars(args)  # is a dictionary
        config = zip(['train', 'val', 'test'], [1000, 200, 200])
        for mode, samples in config:
            dargs['data_mode'] = mode
            dargs['n_samples'] = samples
            prepare_data(dargs)
