import cv2
import glob
import numpy as np
from utils.local_utils import detect_lp
from keras.models import load_model

# https://github.com/quangnhat185/Plate_detect_and_recognize/blob/master/%5BPart%201%5DLicense_plate_detection.ipynb
def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


# https://github.com/quangnhat185/Plate_detect_and_recognize/blob/master/%5BPart%201%5DLicense_plate_detection.ipynb
def get_plate(image_path, wpod, Dmax=608, Dmin=256):
    wpod_net = wpod
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor


# fonction pour reconstruire les images modifie par "preprocess_image" et "get_plate"
def save_img(im, path_save):
    img = im
    min_val, max_val = img.min(), img.max()
    img = 255.0 * (img - min_val) / (max_val - min_val)
    img = img.astype(np.uint8)
    cv2.imwrite(path_save, img[:, :, ::-1])


# fonction pour charge le modele prédéfini et extraire les matricules
def extract_mat(path_cars, path_save_mat):
# charge le modele
    path_model = "./model/model.h5"
    try:
        extract_model = load_model(path_model, compile=False)
        print("Loading model successfully...")
    except:
        print("can't load the model")
        exit

# extraire les matricules
    im_path = path_cars + "/*.jpg"
    image_paths = sorted(glob.glob(im_path), key=len)
    for i, im in enumerate(image_paths):
        try:
            LpImg, _ = get_plate(im, extract_model)
            path = path_save_mat + "/" + str(i) + ".jpg"
            save_img(LpImg[0], path)
            print(f"image {i} saved!!")
        except:
            print(f"can't extract from image {i}")
