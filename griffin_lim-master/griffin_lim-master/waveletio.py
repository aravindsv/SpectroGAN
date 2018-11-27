import pywt
import scipy.io.wavfile as wio
import numpy as np

def audio_to_wavelet(wavfile, filename):

    rate, in_file = wio.read(wavfile)
    cA, cD = pywt.dwt(in_file, 'db2')

    out_data = pywt.idwt(cA, cD, 'db2')
    #out_file = (filename, rate, out_data)

    out_data = np.asarray(out_data, dtype=np.int16)
    wio.write(filename, rate, out_data)
    return cA, cD



if __name__ == '__main__':
    audio_to_wavelet("out.wav","wavetest.wav")
