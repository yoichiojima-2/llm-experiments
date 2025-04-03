from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    opt = parser.add_argument
    opt("--model", "-m", type=str, default="4o-mini")
    opt("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()
