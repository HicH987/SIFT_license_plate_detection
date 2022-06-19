## Lien des codes pri sur internet

- ./model/model.h5: https://github.com/quangnhat185/Plate_detect_and_recognize/blob/master/wpod-net.h5
  
- ./utils/local_utils.py: https://github.com/quangnhat185/Plate_detect_and_recognize/blob/master/local_utils.py


- ./extract_mat.py : 
  - fonction `preprocess_image` : https://github.com/quangnhat185/Plate_detect_and_recognize/blob/master/%5BPart%201%5DLicense_plate_detection.ipynb

  - fonction `get_plate`: https://github.com/quangnhat185/Plate_detect_and_recognize/blob/master/%5BPart%201%5DLicense_plate_detection.ipynb

## Description des fichiers du projet

- ### dataset
  
  - ./dataset/authorized/car: contient les voitures autorisées .
  - ./dataset/authorized/mat: contient les matricules extrait des voitures autorisées .
  - ./dataset/query/car: contient les voitures requetes
  - ./dataset/query/mat: contient les matricules extrait des voitures requetes.

- ### modele
  
  - ./model/model.h5: le modele pre-entraine a l'extraction de matricule.

- ### utils
  
  - ./utils/local_utils.py: contient des fonctions utile pour l'extraction des matricules.
  
- ### les fichiers restent
  
  - ./extract_mat.py: contient les fonctions d'extraction des matricules
  - ./notebook.ipynb: fichier jupyter qui contient le code demander.
