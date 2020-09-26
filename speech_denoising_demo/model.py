import logging as log
from abc import ABC, abstractmethod
import sys
import os
import numpy as np

from .features import calcFeat, calcSpec, spec2sig
from openvino.inference_engine import IENetwork, IECore

class SpeechDenoiser(ABC):
     @staticmethod
     def create(args):
        if args.model == 'nsnet2-20ms-baseline.xml':
            return DeepNoiseSuppression(args.model, args.device)
        else:
            raise Exception('Error: wrong name')

     @abstractmethod
     def denoise(self, data):
         '''Perform Noise Suppression'''

class DeepNoiseSuppression(SpeechDenoiser):
    
    def __init__(self, model, device):
        log.basicConfig(format='[ %(levelname)s ] %(message)s', level= log.INFO, stream=sys.stdout)
        # Plugin initialization
        ie = IECore()
        log.info("Loading network")
        net = ie.read_network(model, os.path.splitext(model)[0] + ".bin")
        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name= device)

    def denoise(self, data):
        self.cfg = {
            'winlen'   : 0.02,
            'hopfrac'  : 0.5,
            'fs'       : 16000,
            'mingain'  : -80,
            'feattype' : 'LogPow'
        }
        self.frameShift = float(self.cfg['winlen'])* float(self.cfg["hopfrac"])
        self.fs = int(self.cfg['fs'])
        self.Nfft = int(float(self.cfg['winlen'])*self.fs)
        self.mingain = 10**(self.cfg['mingain']/20)
        """Enhance a single Audio signal."""
        log.info("Calculate features for audio signal")
        inputSpec = calcSpec(data, self.cfg)
        inputFeature = calcFeat(inputSpec, self.cfg)
        inputFeature = np.expand_dims(np.transpose(inputFeature), axis=0)

        log.info("Starting inference")
        out = self.exec_net.infer({'input': inputFeature})
        log.info("Inference successfull")
        log.info("Processing output")
        res = out['Sigmoid_31'][0]
        Gain = np.transpose(res)
        Gain = np.clip(Gain, a_min=self.mingain, a_max=1.0)
        outSpec = inputSpec * Gain
        # go back to time domain
        out = spec2sig(outSpec, self.cfg)
        log.info("Processing sucessfull")
        return out
