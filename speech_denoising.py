import sys
import argparse
sys.path.insert(0, 'src')
from speech_denoising_demo.model import SpeechDenoiser
import logging as log
import soundfile as sf

def build_argparser():
    parser = argparse.ArgumentParser(description='Speech denoising demo', add_help=False)
    parser.add_argument('-h', '--help', action='help', default = argparse.SUPPRESS,
                        help = 'Show this help message and exit.')
    parser.add_argument('-m', '--model', type=str, required = True,
                        help = 'Required. Path to model (.xml file)')
    parser.add_argument('-i', '--input', type = str, required = True,
                        help = 'Required. Path to audiofile with noisy speech.')
    parser.add_argument('-o', '--output', type = str, required = False, default = './',
                        help = 'Optional. Path to output dir.')
    parser.add_argument('-d', '--device', type=str, default='CPU',
                        help = 'Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                             'acceptable. The demo will look for a suitable plugin for the device specified. '
                             'Default value is CPU')

    args = parser.parse_args()
    return args

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = vars(build_argparser())
    args['name'] = 'DNS'

    log.info("Creating SpeechDenoising Model")
    model = SpeechDenoiser.create(args)
    
    log.info("Reading input data")
    if '.wav' in args['input']:
        # input is single .wav file
        sound, fs = sf.read(args['input'])
        print(sound.shape)

    log.info("Inference for input")
    out = model.denoise(sound)

    log.info("Save the result")
    sf.write(args['output'] + "/result.wav", out, fs)


if __name__ == '__main__':
    sys.exit(main() or 0)
