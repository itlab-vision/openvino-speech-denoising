#!/usr/bin/env python3
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from enhance_onnx import NSnet2Enhancer
from openvino.inference_engine import IECore
import featurelib
"""
    Inference script for NSnet2 baseline.
"""
class Model:
    def __init__(self, args):
        self.core = IECore()
        self.network = self.core.read_network(args.model, args.model.replace(".xml", ".bin"))
        self.input_blob = next(iter(self.network.input_info))
        outputs = list(self.network.outputs.keys())
        for key in outputs:
            print(self.network.outputs[key].shape)
        self.output_blob = outputs[2]
        self.exec_net = self.core.load_network(self.network, device_name="CPU")
        self.cfg = {
            'winlen'   : 0.02,
            'hopfrac'  : 0.5,
            'fs'       : 16000,
            'mingain'  : -80,
            'feattype' : 'LogPow'
        }
        self.mingain = 10**(self.cfg['mingain']/20)
    
    def infer(self, sigIn):
        inputSpec = featurelib.calcSpec(sigIn, self.cfg)
        inputFeature = featurelib.calcFeat(inputSpec, self.cfg)
        # shape: [batch x time x freq]
        inputFeature = np.expand_dims(np.transpose(inputFeature), axis=0)
        outSig = self.exec_net.infer(inputs={self.input_blob: inputFeature})
        out = outSig[self.output_blob][0]
        print("Shape of out = ", out.shape)
        Gain = np.transpose(out)
        print(self.mingain, type(self.mingain))
        Gain = np.clip(Gain, a_min=self.mingain, a_max=1.0)
        outSpec = inputSpec * Gain
        # go back to time domain
        sigOut = featurelib.spec2sig(outSpec, self.cfg)
        return sigOut


def main(args):
    # check input path
    inPath = Path(args.input).resolve()
    assert inPath.exists()

    # Create the enhancer IR network
    enc = Model(args)
    #enhancer = NSnet2Enhancer(modelfile=args.model)

    # get modelname
    modelname = Path(args.model).stem

    if inPath.is_file() and inPath.suffix == '.wav':
        # input is single .wav file
        sigIn, fs = sf.read(str(inPath))
        if len(sigIn.shape) > 1:
            sigIn = sigIn[:,0]

        outSig = enc.infer(sigIn)
        outname = './{:s}_{:s}.wav'.format(inPath.stem, modelname)
        if args.output:
            # write in given dir
            outdir = Path(args.output)
            outdir.mkdir(exist_ok=True)
            outpath = outdir.joinpath(outname)
        else:
            # write in current work dir
            outpath = Path(outname)

        print('Writing output to:', str(outpath))
        sf.write(outpath.resolve(), outSig, fs)

    elif inPath.is_dir():
        # input is directory
        if args.output:
            # full provided path
            outdir = Path(args.output).resolve()
        else:
            outdir = inPath.parent.joinpath(modelname).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        print('Writing output to:', str(outdir))

        fpaths = list(inPath.glob('*.wav'))
        for ii, path in enumerate(fpaths):
            print(f"Processing file [{ii+1}/{len(fpaths)}]")
            sigIn, fs = sf.read(str(path))
            if len(sigIn.shape) > 1:
                sigIn = sigIn[:,0]
            print(sigIn.shape, fs)
            outSig = enc.infer(sigIn)
            print(outSig.shape, type(outSig))
            outpath = outdir / path.name

            sf.write(str(outpath), outSig, fs)

    else:
        raise ValueError("Invalid input path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, help="Path to IR model.", default='nsnet2-20ms-baseline.onnx')
    parser.add_argument("-i", "--input", type=str, help="Path to noisy speech wav file or directory.")
    parser.add_argument("-o", "--output", type=str, help="Optional output directory.", required=False)
    args = parser.parse_args()

    main(args)