import gdown,os
from pathlib import Path
import requests
import zipfile,shutil,bz2

def get_deepface_home():
    return str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))

def initialize_folder():
    home = get_deepface_home()

    if not os.path.exists(home + "/.deepface"):
        os.makedirs(home + "/.deepface")
        print("Directory ", home, "/.deepface created")

    if not os.path.exists(home + "/.deepface/weights"):
        os.makedirs(home + "/.deepface/weights")
        print("Directory ", home, "/.deepface/weights created")

def copy_weights_dlib():
    initialize_folder()
    home = get_deepface_home()

    file_name = "shape_predictor_5_face_landmarks.dat.bz2"
    print(f"{file_name} is going to be copied")
    output = f"{home}/.deepface/weights"
    input1 = './model/shape_predictor_5_face_landmarks.dat.bz2'
    shutil.copy(src=input1,dst=output)

    

    downloaded_file_name = output + '/' + file_name
    extracted_file_name = output + '/' + 'shape_predictor_5_face_landmarks.dat'
    with open(downloaded_file_name, "rb") as file:
        decompressed_data = bz2.decompress(file.read())
    with open(extracted_file_name, "wb") as extracted_file:
        extracted_file.write(decompressed_data)
    print(f"Extracted {extracted_file_name} successfully!")
    
    os.remove(downloaded_file_name)
    print(f"Deleted {downloaded_file_name}")


    file_name = 'dlib_face_recognition_resnet_model_v1.dat.bz2'
    print(print(f"{file_name} is going to be copied"))
    input2 = './model/dlib_face_recognition_resnet_model_v1.dat.bz2'
    shutil.copy(src=input2,dst=output)

    downloaded_file_name = output + '/' + file_name
    extracted_file_name = output + '/' + 'dlib_face_recognition_resnet_model_v1.dat'
    with open(downloaded_file_name, "rb") as file:
        decompressed_data = bz2.decompress(file.read())
    with open(extracted_file_name, "wb") as extracted_file:
        extracted_file.write(decompressed_data)
    print(f"Extracted {extracted_file_name} successfully!")
    
    os.remove(downloaded_file_name)
    print(f"Deleted {downloaded_file_name}")


copy_weights_dlib()