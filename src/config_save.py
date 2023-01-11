from config import configure_parse, configure_save, check_dependency


def main():
    args = configure_parse(load_config=False)
    check_dependency(args)
    if args.config is not None:
        file_path = args.config
        args_dict = vars(args)
        del args_dict['config']
        configure_save(file_path, args_dict)


if __name__ == '__main__':
    main()