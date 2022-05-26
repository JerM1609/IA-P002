import pandas as pd
import numpy as np
from PIL import Image
import os
from natsort import natsorted

def get_data(src_dir):
    """
    get_data

    :param src_dir: Directorio origen para leer las imágenes.
    :return data: Lista de Dataframes con la matriz de colores y pixeles de cada imagen.
    """
    data = []

    for train_img in natsorted(os.listdir(src_dir)):
        image_path = f"{src_dir}/{train_img}"
        img = Image.open(image_path)
        newsize = (100, 100)
        img = img.resize(newsize)
        matrix = np.asarray(img)
        matrix = [list(matrix[i][j]) + [i, j] for i in range(len(matrix)) for j in range(len(matrix[i]))]
        matrix = pd.DataFrame(matrix, columns=['R','G','B','i','j'])
        data.append((train_img.replace(".jpg", "") ,matrix))
    return data