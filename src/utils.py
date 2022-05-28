import pandas as pd
import numpy as np
from PIL import Image
import os
import pywt
from natsort import natsorted
from sklearn.neighbors import NearestNeighbors
import  matplotlib.pyplot as plt

def get_data(src_dir, width=100, height=100):
    """
    get_data

    :param src_dir: Directorio origen para leer las imágenes.
    :return data: Lista de Dataframes con la matriz de colores y pixeles de cada imagen.
    """
    data = []

    for train_img in natsorted(os.listdir(src_dir)):
        image_path = f"{src_dir}/{train_img}"
        img = Image.open(image_path)
        newsize = (width, height)
        img = img.resize(newsize)
        matrix = np.asarray(img)
        matrix = [list(matrix[i][j]) + [i, j] for i in range(len(matrix)) for j in range(len(matrix[i]))]
        matrix = pd.DataFrame(matrix, columns=['R','G','B','i','j'])
        data.append((train_img.replace(".jpg", "") ,matrix))
    return data

def extract_tumor_to_image(dataframe, cluster_tumor):
    i_max = dataframe['i'].max()+1
    j_max = dataframe['j'].max()+1
    array = np.zeros((i_max,j_max,3), dtype=np.uint8)
    for index in range(len(dataframe)):
        i = dataframe.iloc[index]['i']
        j = dataframe.iloc[index]['j']
        cluster = dataframe.iloc[index]['cluster']
        color = dataframe.iloc[index]['R']
        if cluster != cluster_tumor:
            array[i,j] = [0, 0, 0]
        else:
            array[i,j] = [color, color, color]
    
    return array


def get_vector_from_tumor(tumor_imagen, iterations):
    """
    get_vector_from_image obtiene el vector característico de la imagen image

    :param image: Imagen en formato vector.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return LL: Vector característico sin la compresión a 1D.
    :return LL.flatten(): Vector característico en 1D.
    """
    LL, (LH, HL, HH) = pywt.dwt2(tumor_imagen, 'haar')
    for _ in range(iterations - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL.flatten()



def get_tumors_wavelet(src_dir, iterations):
    """
    get_data

    :param src_dir: Directorio origen para leer las imágenes.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return np.asarray(x): Vector con los vectores característicos de las imágenes en 1D.
    :return np.asarray(y): Vector con los labels correspondientes a los vectores característicos.
    """
    x = []
    image_names = []

    for train_img in natsorted(os.listdir(src_dir)):
        image_path = f"{src_dir}/{train_img}"
        img = Image.open(image_path)
        fv = get_vector_from_tumor(img, iterations)
        x.append(fv)
        image_names.append(train_img.replace('png','jpg'))
    return np.asarray(x), np.asarray(image_names)


def get_similar_tumors(X, image_names, index_to_test, n_neighbors):
    n_neighbors += 1
    neigh = NearestNeighbors(n_neighbors=n_neighbors,algorithm='kd_tree')
    neigh.fit(X)
    nombre_imagen_original = image_names[index_to_test]

    similares = neigh.kneighbors([X[index_to_test]], return_distance=False)
    similares = image_names[similares][0][1:]

    fig, ax= plt.subplots(1,n_neighbors, figsize=(20,20))

    im = plt.imread('data/'+nombre_imagen_original)
    ax[0].imshow(im, extent=[0, 100, 0, 100])
    ax[0].axis('off')
    ax[0].set_title('TUMOR ORIGINAL')

    for i in range(1,n_neighbors):
        im = plt.imread('data/'+similares[i-1])
        ax[i].imshow(im,  extent=[0, 100, 0, 100])
        ax[i].axis('off')
        ax[i].set_title(f"TUMOR SIMILAR #{i}")

    plt.show()