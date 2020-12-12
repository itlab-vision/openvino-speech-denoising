import logging as log
import os
import sys
import time
import argparse
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from openvino.inference_engine import IECore
from speech_denoising_demo.model import SpeechDenoiser


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Audiofile to process. Default value is "mic" to work with default microphone')
    args.add_argument('-o', '--output', type=str, default='./',
                      help='Optional. Path to output dir.')
    args.add_argument('-d', '--device', type=str, default='CPU',
                      help='Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                      'acceptable. The demo will look for a suitable plugin for the device specified. '
                      'Default value is CPU')
    args.add_argument('-no_play',  action='store_true',
                      help='Optional. Do not play inference results.')

    return parser

# def preprocess_audio(input)


def main():
    """ Main function. """
    log.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info('Create denoiser')
    denoiser = SpeechDenoiser.create(args)
    fs = 16000  # default sample rate for input
    if args.input == 'mic':
        log.info('Start recording')
        duration = 10  # seconds
        data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        log.info('Saving recording')
        sf.write('recording.wav', data, fs)
        log.info('Recording was written to ' +
                 str(os.path.abspath(os.getcwd())) + '\\recording.wav')
    else:
        log.info('Читаем запись звука')
        data, fs = sf.read(args.input)
        #log.info('Playing record')
        #sd.play(data, fs)

    log.info('Семплы:')
    print(data)
    log.info(
        'Каждый семпл представляет собой амплитуду волны в определенном временном интервале')
    log.info('Частота дискретизации:')
    print(fs)
    log.info(
        'Частота дискретизации - количество семплов за определенный фиксированный промежуток времени')

    log.info('График амплитуды от времени:')
    plt.figure(num='Recording', figsize=(15, 7))
    plt.ylabel('dB')
    librosa.display.waveplot(data, fs)
    plt.show()

    # запуск подавителя
    res = denoiser.denoise(data)

    #log.info('Playing result')
    #sd.play(res, fs)

    log.info('Сравнение результатов:')
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].set_title('Input')
    axs[0].set_ylabel('dB')
    librosa.display.waveplot(data, fs, ax=axs[0])
    axs[1].set_title('Output')
    axs[1].set_ylabel('dB')
    librosa.display.waveplot(res, fs, ax=axs[1])
    fig.suptitle("Results")
    plt.show()

    status = sd.wait()

    log.info('Saving result')
    sf.write('res.wav', res, fs)
    log.info('Result was written to ' + str(os.path.abspath(os.getcwd())
                                            ) + '\\res.wav')  # TODO: add output parameter
    # at least two cases : 1) upload audio, process it an than play. 2) record, process, play


if __name__ == '__main__':
    sys.exit(main() or 0)
