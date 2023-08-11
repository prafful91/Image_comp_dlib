from fastapi import FastAPI, File, UploadFile
import shutil
import os
from io import BytesIO
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image,ImageDraw
from datetime import datetime

app = FastAPI()

@app.get("/")
async def index():
    return "Hi"

@app.post("/compare_images/")
async def compare_images(files: list[UploadFile] = File(...)):
    if len(files) != 2:
        return {"error": "Please provide exactly two image files."}
    
    # Save the images temporarily to a folder
    image_paths = []
    for idx, file in enumerate(files):
        contents = await file.read()
        #contents = cv2.imread(BytesIO(contents))
        image_path = fr"D:\Object_detection\temp_image\image{idx}.jpg"
        print(f"\n\n{image_path}\n\n")
        with open(image_path, "wb") as f:
            f.write(contents)
        #cv2.imwrite(image_path, contents)
        image_paths.append(image_path)
    
    # Perform image comparison
    try:
        print('Image comparison------------')
        print(image_paths)
        result = DeepFace.verify(image_paths[0], image_paths[1],enforce_detection=False)
        
        similarity = result["verified"]

        print(similarity)

        print('completed\n\n\n')
    except Exception as e:
        return {"error": f"Error in image comparison: {str(e)}"}

    # Clean up the temporary images
    for image_path in image_paths:
        os.remove(image_path)

    return {"similarity": str(similarity)}

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os
import base64



# Mount a directory containing static files (e.g., images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2Templates
templates = Jinja2Templates(directory="templates")

# HTML template for displaying images



@app.post("/upload/")
async def upload_images(request: Request, image1: UploadFile = File(...), image2: UploadFile = File(...)):
    # Save uploaded images to the 'static' directory
    start_time = datetime.now()
    image1_path = os.path.join("static", image1.filename)
    image2_path = os.path.join("static", image2.filename)

    with open(image1_path, "wb") as f1, open(image2_path, "wb") as f2:
        shutil.copyfileobj(image1.file, f1)
        shutil.copyfileobj(image2.file, f2)

    
    result = DeepFace.verify(image1_path,image2_path,
                         enforce_detection=True,
                         model_name='Dlib',
                         detector_backend='dlib',
                         distance_metric='euclidean')
        
    similarity = result["verified"]

    print(similarity)

    print('completed\n\n\n') 

    print(result)   

    cord_img1 = result['facial_areas']['img1']
    cord_img2 = result['facial_areas']['img2']

    x1 = cord_img1['x']
    y1 = cord_img1['y']
    w1 = cord_img1['w']
    h1 = cord_img1['h']


    x2 = cord_img2['x']
    y2 = cord_img2['y']
    w2 = cord_img2['w']
    h2 = cord_img2['h']
    
    RESIZE_SHAPE = (200,200)



    cv2_img1,cv2_img2 = Image.open(image1_path),Image.open(image2_path)

    bottom_1 = (x1 + w1, y1 + h1)
    draw = ImageDraw.Draw(cv2_img1)
    draw.rectangle([(x1,y1),bottom_1], outline="red", width=3)

    bottom_2 = (x2 + w2, y2 + h2)
    draw2 = ImageDraw.Draw(cv2_img2)
    draw2.rectangle([(x2,y2),bottom_2], outline="green", width=3)
    
    bb_img1_path = './static/op3.jpg'
    bb_img2_path = './static/op4.jpg'

    cv2_img1 = cv2_img1.resize(RESIZE_SHAPE)
    cv2_img2 = cv2_img2.resize(RESIZE_SHAPE)

    cv2_img1.save(bb_img1_path)
    cv2_img2.save(bb_img2_path)

    print(cord_img2,cord_img1)

    cv2_img1,cv2_img2 = Image.open(image1_path),Image.open(image2_path)

    cv2_img1 = cv2_img1.crop((x1, y1, x1 + w1, y1 + h1))
    cv2_img2 = cv2_img2.crop((x2, y2, x2 + w2, y2 + h2))
    
    cv2_img1 = cv2_img1.resize(RESIZE_SHAPE)
    cv2_img2 = cv2_img2.resize(RESIZE_SHAPE)

    cropped_img1_path = './static/op1.jpg'
    cropped_img2_path = './static/op2.jpg'

    cv2_img1.save(cropped_img1_path)
    cv2_img2.save(cropped_img2_path)




    # Convert images to base64 strings for display
    with open(cropped_img1_path, "rb") as f1, open(cropped_img2_path, "rb") as f2:
        image1_base64 = base64.b64encode(f1.read()).decode("utf-8")
        image2_base64 = base64.b64encode(f2.read()).decode("utf-8")

    with open(bb_img1_path, "rb") as f1, open(bb_img2_path, "rb") as f2:
        image1_bb_base64 = base64.b64encode(f1.read()).decode("utf-8")
        image2_bb_base64 = base64.b64encode(f2.read()).decode("utf-8")    

    for image_path in os.listdir('./static'):

        if image_path.endswith(".txt"):
            continue
        del_path = './static/' + image_path

        if os.path.exists(del_path):
            os.remove(del_path)


    time_diff = datetime.now() - start_time
    total_seconds = round(time_diff.total_seconds(),1)

    return templates.TemplateResponse(
        "image_template.html",
        {
          "request": request,
          "image1": f"data:image/jpeg;base64,{image1_base64}", 
          "image1_bb": f"data:image/jpeg;base64,{image1_bb_base64}", 
          "image2": f"data:image/jpeg;base64,{image2_base64}",
          "image2_bb": f"data:image/jpeg;base64,{image2_bb_base64}", 
          'result':str(similarity),
          'total_seconds':total_seconds
          }
    )

    

@app.get('/upload_images/')
async def read_item(request:Request):
    return templates.TemplateResponse(
            "form_template.html",
            {"request": request}
        )