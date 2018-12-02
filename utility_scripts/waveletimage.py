import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pywt
import pywt.data

import scipy.io.wavfile as wio

def waveletimage(filename):
    # Load image
    rate, in_file = wio.read(filename)
    original = in_file
    image = pywt.data.camera()
    print(image.shape)
    print(original.shape)
    #print(original[2])
    print(np.sqrt(original.shape))
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
    return 0

if __name__ == '__main__':
    waveletimage("out.wav")
