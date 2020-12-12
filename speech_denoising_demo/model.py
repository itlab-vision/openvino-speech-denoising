import logging as log
from abc import ABC, abstractmethod
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from .features import calcFeat, calcSpec, spec2sig
from openvino.inference_engine import IECore


class SpeechDenoiser(ABC):
    @staticmethod
    def create(args):
        return DeepNoiseSuppression(args.model, args.device)

    @abstractmethod
    def denoise(self, data):
        '''Perform Noise Suppression'''


class DeepNoiseSuppression(SpeechDenoiser):
    def __init__(self, model, device):
        log.basicConfig(format='[ %(levelname)s ] %(message)s',
                        level=log.INFO, stream=sys.stdout)
        self.cfg = {
            'winlen': 0.02,
            'hopfrac': 0.5,
            'fs': 16000,
            'mingain': -80,
            'feattype': 'LogPow'
        }

        # минимальные значения для выхода нейросети
        self.mingain = 10**(self.cfg['mingain']/20)

        # Plugin initialization
        ie = IECore()
        log.info("Loading network")
        net = ie.read_network(model, os.path.splitext(model)[0] + ".bin")
        self.input_blob = next(iter(net.input_info))  # ?
        self.output_blob = 'output'
        assert len(net.input_info) == 1, "One input is expected"

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=device)

    def preprocessing(self, data):
        # получаем спектр (образ Фурье)
        self.inputSpec = calcSpec(data, self.cfg)

        log.info('Входной спектр после применения STFT:')
        print(self.inputSpec)

        # получаем логарифмический спектр мощности
        inputFeature = calcFeat(self.inputSpec, self.cfg)

        log.info('Логарифмический спектр мощности:')
        print(inputFeature)

        log.info('График обычного спектра мощности:')
        plt.figure(num='Spectr power ordinary', figsize=(15, 7))
        librosa.display.specshow(librosa.amplitude_to_db(
            abs(self.inputSpec)),  x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()

        log.info('График логарифмического спектра мощности:')
        plt.figure(num='Spectr power log', figsize=(15, 7))
        librosa.display.specshow(inputFeature,  x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()

        # shape: [batch x time x freq]
        inputFeature = np.expand_dims(np.transpose(inputFeature), axis=0)

        return inputFeature

    def postprocessing(self, out):
        # Gain - фильтр, выход нейросети, восстанавливает амплитуду речи
        Gain = np.transpose(out)

        log.info('Выход нейросети (фильтр):')
        print(Gain)

        # обрезаем значения, поскольку мы стремимся только к уменьшению аддитивного шума a_max=1.0
        Gain = np.clip(Gain, a_min=self.mingain, a_max=1.0)

        # применяя фильтр к входному спектру (образ Фурье), избавляемся от шума
        outSpec = np.expand_dims(self.inputSpec, axis=2) * Gain

        log.info('Входной спектр после применения фильтра:')
        print(outSpec)

        # преобразуем обратно в массив амплитуд звукового сигнала
        out = spec2sig(outSpec, self.cfg)

        log.info('Результирующие семплы:')
        print(out)

        return out

    def denoise(self, data):
        log.info("Preprocessing input")

        # получаем спектр мощности для входа нейросети
        inputFeature = self.preprocessing(data)

        # Calculate network output
        log.info("Starting inference")
        out = self.exec_net.infer({self.input_blob: inputFeature})[
            self.output_blob]
        log.info("Inference successfull")
        log.info("Output shape is " + str(out.shape))
        log.info("Postprocessing output")

        # обрабатываем выход нейросети (фильтр) и конвертируем выходной спектр в звук
        result = self.postprocessing(out)
        return result
