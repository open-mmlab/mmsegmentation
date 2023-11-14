from mmengine.config import Config
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Resolve and print config')
    parser.add_argument('config', help='the config to resolve', type=str)
    args = parser.parse_args()
    config = Config.fromfile(args.config).to_dict()
    config = json.dumps(config, indent=2)
    print(config)


if __name__ == "__main__":
    main()