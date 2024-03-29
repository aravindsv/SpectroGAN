import numpy as np
import argparse
from time import time
from sklearn.decomposition import PCA
import scipy.io.wavfile as wio
import argparse
import pywt

def get_weights(vector, mean, eigenvectors):
    #input data vector and eigenvectors as numpy array
    weights = []
    vector = np.copy(vector)
    vector = vector - mean
    for ev in eigenvectors:
        weight_entry = np.dot(vector, ev)
        #print(ev)
        weights.append(weight_entry)
    #print("Weights: ", weights)
    return weights;

def do_PCA(matrix, n_components):
    #make sure that matrix is a numpy array folks
    print("Extracting the top %d eigenvectors from %d input vectors"
      % (n_components, matrix.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(matrix)
    #pca is a proprietary data type from scikit
    #might be worth exporting somehow
    print("done in %0.3fs" % (time() - t0))
    print("variance explained: %d" % (np.sum(pca.explained_variance_ratio_)))

    np.savez('pcaeig_{}components_speech.npz'.format(n_components), mean=pca.mean_, components=pca.components_)
    return pca;

def Alt_PCA_recon(X, pca, nComp):
    mu = pca.mean_
    X = X-np.mean(X)
    print("Reconstructing using %d eigenvectors" % (nComp))
    Xhat = np.dot(X, pca.components_[:nComp,:])
    Xhat += mu
    #  print("PCA 1: ", Xhat)
    return Xhat;

def PCA_recon(weights, mean, eigenvectors):
    recon_vector = np.matmul(weights, eigenvectors)
    recon_vector += mean
    #print("PCA 2: ", recon_vector)
    return recon_vector;

def wavelet_recon(data):
    cA = data[:len(data)//2]
    cD = data[len(data)//2:]
    cA, cD = cA[:-1], cD[:-1]
    out_data = pywt.idwt(cA, cD, 'db2')
    return out_data;

def wavelet_load(filename):
        rate, wav = wio.read(filename)
        cA, cD = pywt.dwt(wav, 'db2')
        cA, cD = cA[:-1], cD[:-1]
        waveletc = np.hstack([cA, cD])
        return waveletc, rate;

test_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 2, 2]])
#test_vector = np.array([1., 4., 7.])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_matrix', help='npy file with audio matrix of all training vectors')
    parser.add_argument('-n', '--num-components', help='Number of components to use for PCA', type=int, default=100)
    args = parser.parse_args()

    rate, in_file4 = wio.read('birdmono021.wav')
    matrix = np.load(args.audio_matrix)
    print(test_matrix.shape, matrix.shape)
    pca = do_PCA(matrix, args.num_components)
    #print("Eigenvectors: \n", pca.components_, "\n")
    weights = get_weights(in_file4, pca.mean_, pca.components_)
    import matplotlib.pyplot as plt
    plt.plot(weights)
    plt.show()
    out_data = PCA_recon(weights, pca.mean_, pca.components_)
#    out_data = wavelet_recon(out_data)
#    out_data = pca.transform(in_file4.reshape(-1, 1))
    out_data = out_data.astype(np.int16)
    wio.write("pcatest.wav", rate, out_data)
    np.savez('pcadata.npz', mean = pca.mean_, components = pca.components_)
