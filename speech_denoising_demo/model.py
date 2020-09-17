from abc import ABC, abstractmethod
import sys, os
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import features

class SpeechDenoiser(ABC):
     @staticmethod
     def create(args):
        if args['name'] == 'DNS':
            return DNS(args['xml'])
        else:
            raise Exception('Error: wrong name')

     @abstractmethod
     def denoise(self, img, ID):
         '''Perform Noise Suppression'''

class DNS(SpeechDenoiser):
    def __init__(self, model):
        ''''''

    def denoise(self, sigIn, inFs):
        ''''''