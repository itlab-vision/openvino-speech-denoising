import sys
import argparse
sys.path.insert(0, 'src')
import model as m
import soundfile as sf

def build_argparser():
    parser = argparse.ArgumentParser(description='Speech denoising demo')
    parser.add_argument('-h', '--help', action='help', default = argparse.SUPPRESS,
                        help='Show this help message and exit.')
    parser.add_argument('-i', '--input', type = str, required=True,
                        help = 'Required. Path to audiofile with noisy speech.')
    parser.add_argument('-o', '--output', type = str, required=True, default = './',
                        help = 'Optional. Path to output dir.')
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help='Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                             'acceptable. The demo will look for a suitable plugin for the device specified. '
                             'Default value is CPU',)

    args = parser.parse_args()
    return args

def main():
    args = build_argparser().parse_args()

if __name__ == '__main__':
    sys.exit(main() or 0)