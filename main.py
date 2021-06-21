import argparse
import sys
import torchlight
from torchlight.io import import_class


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    #processors['demo'] = import_class('processor.demo.Demo')

    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])
    p.start()
