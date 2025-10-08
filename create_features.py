import os
import sys
import argparse
import _pickle as pickle

import cv2
import numpy as np
from sklearn.cluster import KMeans


class DenseDetector():
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        # Create a dense feature detector
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints


class SIFTExtractor():
    def __init__(self):
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps):
        if image is None:
            print("Not a valid image")
            raise TypeError

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = self.extractor.compute(gray_image, kps)
        if des is not None:
            des = des.astype(np.float32) 
        return kps, des


# Vector quantization
class Quantizer(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.extractor = SIFTExtractor()
        self.num_clusters = num_clusters
        self.num_retries = 10

    def quantize(self, datapoints):

        datapoints = np.array(datapoints, dtype=np.float32)
        
        # Create KMeans object
        kmeans = KMeans(self.num_clusters,
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)

        # Run KMeans on the datapoints
        res = kmeans.fit(datapoints)

        # Extract the centroids of those clusters
        centroids = res.cluster_centers_

        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

            # Extract feature vector from the image

    def get_feature_vector(self, img, kmeans, centroids):
        kps = DenseDetector().detect(img)
        kps, fvs = self.extractor.compute(img, kps)
        labels = kmeans.predict(fvs)
        fv = np.zeros(self.num_clusters)

        for i, item in enumerate(fvs):
            fv[labels[i]] += 1

        fv_image = np.reshape(fv, ((1, fv.shape[0])))
        return self.normalize(fv_image)


class FeatureExtractor(object):
    def extract_image_features(self, img):
        # Dense feature detector
        kps = DenseDetector().detect(img)

        # SIFT feature extractor
        kps, fvs = SIFTExtractor().compute(img, kps)

        return fvs

        # Extract the centroids from the feature points

    def get_centroids(self, input_map, num_samples_to_fit=10):
        kps_all = []

        count = 0
        cur_label = ''

        for item in input_map:
            if count >= num_samples_to_fit:
                if cur_label != item['label']:
                    count = 0
                else:
                    continue

            count += 1

            cur_label = item['label']
            img = cv2.imread(item['image'])
            
            # <<< AÑADIR VERIFICACIÓN AQUÍ >>>
            if img is None:
                print(f"!!! ERROR: No se pudo leer la imagen: {item['image']}. Saltando.")
                continue # Saltar esta iteración y pasar a la siguiente imagen

            img = resize_to_size(img, 150) # Línea 116 original
            
            fvs = self.extract_image_features(img)
            kps_all.extend(fvs)

        kmeans, centroids = Quantizer().quantize(kps_all)
        return kmeans, centroids

    def get_feature_vector(self, img, kmeans, centroids):
        return Quantizer().get_feature_vector(img, kmeans, centroids)



def extract_feature_map(input_map, kmeans, centroids):
    feature_map = []

    for item in input_map:
        temp_dict = {}
        temp_dict['label'] = item['label']

        print("Extracting features for", item['image'])
        img = cv2.imread(item['image'])
        img = resize_to_size(img, 150)

        temp_dict['feature_vector'] = FeatureExtractor().get_feature_vector(img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)

    return feature_map


def resize_to_size(input_image, new_size=150):
    h, w = input_image.shape[0], input_image.shape[1]
    ds_factor = new_size / float(h)

    if w < h:
        ds_factor = new_size / float(w)

    new_size = (int(w * ds_factor), int(h * ds_factor))
    return cv2.resize(input_image, new_size)


def build_arg_parser():
    # ... (Dejar el argparse como está para compatibilidad) ...
    parser = argparse.ArgumentParser(description='Creates features for given images')
    parser.add_argument("--samples", dest="cls", nargs="+", action="append", required=True, \
        help="Folders containing the training images. The first element needs to be the class label.")
    parser.add_argument("--codebook-file", dest='codebook_file', required=True,
        help="Base file name to store the codebook")
    parser.add_argument("--feature-map-file", dest='feature_map_file', required=True, \
        help="Base file name to store the feature map")
    return parser


def load_input_map(label, input_folder):
    combined_data = []
    if not os.path.isdir(input_folder):
        raise IOError("The folder " + input_folder + " doesn't exist")
    
    # Supongamos que esta función recorre la carpeta y encuentra los archivos .jpg
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith(('.jpg', '.jpeg', '.png'))):
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})
    return combined_data

# -------------------------------------------------------------------

# --- MODIFICACIÓN AQUÍ ---
if __name__ == '__main__':
    
    # 1. Intentar analizar argumentos como se hacía originalmente
    try:
        args = build_arg_parser().parse_args()
        
        input_map = []
        for cls in args.cls:
            assert len(cls) >= 2, "Format for classes is `<label> folder`"
            label = cls[0]
            input_map += load_input_map(label, cls[1])
            
        codebook_file = args.codebook_file
        feature_map_file = args.feature_map_file
        
    except SystemExit:
        # Esto ocurre si parse_args falla (ej: faltan argumentos).
        # Si falló, asumimos que el usuario quiere usar la configuración fija.
        
        # --- CONFIGURACIÓN FIJA CON TUS RUTAS Y NOMBRES ---
        print("\n=== ATENCIÓN: Usando configuración de entrenamiento fija ===\n")
        
        BASE_IMAGE_FOLDER = 'images' 
        CODEBOOK_FILE_FIXED = 'codebook.pkl'
        FEATURE_MAP_FILE_FIXED = 'label_encoder.pkl'
        
        # Escanear subcarpetas dentro de 'images' para encontrar clases
        input_map = []
        try:
            for label in os.listdir(BASE_IMAGE_FOLDER):
                folder_path = os.path.join(BASE_IMAGE_FOLDER, label)
                if os.path.isdir(folder_path):
                    print(f"-> Cargando clase '{label}' desde: {folder_path}")
                    input_map += load_input_map(label, folder_path)
            
            if not input_map:
                 raise Exception(f"No se encontraron imágenes en subcarpetas de '{BASE_IMAGE_FOLDER}'.")

        except FileNotFoundError:
             print(f"ERROR: La carpeta '{BASE_IMAGE_FOLDER}' no existe.")
             sys.exit(1)
        except Exception as e:
             print(f"ERROR: {e}")
             sys.exit(1)

        codebook_file = CODEBOOK_FILE_FIXED
        feature_map_file = FEATURE_MAP_FILE_FIXED
        
        
    # --- PROCESO DE ENTRENAMIENTO (COMÚN A AMBOS FLUJOS) ---
    if not input_map:
        print("ERROR: No se cargó ningún dato. Terminando.")
        sys.exit(1)
        
    # Building the codebook
    print("===== Building codebook =====")
    # ... (El resto de la lógica de entrenamiento) ...
    kmeans, centroids = FeatureExtractor().get_centroids(input_map)
    
    if codebook_file:
        with open(codebook_file, 'wb') as f:
            print('kmeans', kmeans)
            print('centroids', centroids)
            pickle.dump((kmeans, centroids), f)
            print(f'Codebook guardado en: {codebook_file}')

    # Input data and labels
    print("===== Building feature map =====")
    feature_map = extract_feature_map(input_map, kmeans, centroids)
    
    if feature_map_file:
        with open(feature_map_file, 'wb') as f:
            pickle.dump(feature_map, f)
            print(f'Feature Map guardado en: {feature_map_file}')
