import argparse

class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SHAMPU main')
        self.parser.add_argument(
            '-a', '--alg_params_json',
            type=str,
            required=True,
            help='Path to the algorithm parameters JSON file'
        )

    def parse(self, program_name):
        return self.parser.parse_args()
