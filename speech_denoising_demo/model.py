from abc import ABC, abstractmethod
import sys, os
import numpy as np
from .features import calcSpec, calcFeat, spec2sig
from openvino.inference_engine import IECore

class SpeechDenoiser(ABC):
    ie_core = IECore()

    @staticmethod
    def create(args):
       if args['name'] == 'DNS':
           return DeepNoiseSuppression(args['model'], args['device'])
       else:
           raise Exception('Error: wrong name')
    @abstractmethod
    def denoise(self, img, ID):
        '''Perform Noise Suppression'''

class DeepNoiseSuppression(SpeechDenoiser):
    def __init__(self, model, device='CPU'):
        print("Parameters for model ....")
        self.cfg = {
            'winlen'   : 0.02,
            'hopfrac'  : 0.5,
            'fs'       : 16000,
            'mingain'  : -80,
            'feattype' : 'LogPow'
        }
        self.mingain = 10**(self.cfg['mingain']/20)

        print("Reading IR for model ....")
        self.network = SpeechDenoiser.ie_core.read_network(model, model.replace(".xml", ".bin"))
        self.input_blob = next(iter(self.network.input_info))
        self.output_blob = 'output'
        assert len(self.network.input_info) == 1, "One input is expected"

        print("Loading network to the plugin...")
        self.exec_net = SpeechDenoiser.ie_core.load_network(network=self.network, device_name=device)
    
    def preprocessing(self, sigIn):
        self.inputSpec = calcSpec(sigIn, self.cfg)
        inputFeature = calcFeat(self.inputSpec, self.cfg)
        # shape: [batch x time x freq]
        inputFeature = np.expand_dims(np.transpose(inputFeature), axis=0)
        return inputFeature

    def postprocessing(self, out):
        Gain = np.transpose(out)
        Gain = np.clip(Gain, a_min=self.mingain, a_max=1.0)
        outSpec = np.expand_dims(self.inputSpec, axis=2) * Gain

        # go back to time domain
        sigOut = spec2sig(outSpec, self.cfg)
        return sigOut

    def denoise(self, sigIn):
        inputFeature = self.preprocessing(sigIn)

        # Calculate network output
        out = self.exec_net.infer({self.input_blob: inputFeature})[self.output_blob]
        # print(out)
        print(out.shape)
        result = self.postprocessing(out)
        return result
