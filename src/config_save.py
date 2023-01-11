from config import configure_parse, configure_save, check_dependency


def main():
    args = configure_parse(load_config=False)
    check_dependency(args)
    if args.config is not None:
        file_path = args.config
        configure_save(file_path, vars(args))


if __name__ == '__main__':
    main()