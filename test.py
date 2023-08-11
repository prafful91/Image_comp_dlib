import dlib
import face_recognition
from PIL import Image, ImageDraw

# Load the shape predictor and face recognition models
shape_predictor_model = "./models/shape_predictor_5_face_landmarks.dat"
recognition_model = "./models/dlib_face_recognition_resnet_model_v1.dat"
sp = dlib.shape_predictor(shape_predictor_model)
facerec = dlib.face_recognition_model_v1(recognition_model)

# Load images of the two faces you want to compare
image_path1 = r"C:\Users\Hp\Downloads\passport1.jpg"
image_path2 = r"C:\Users\Hp\Downloads\pan card.jpg"

image1 = face_recognition.load_image_file(image_path1)
image2 = face_recognition.load_image_file(image_path2)

# Find face landmarks and face encodings
face_landmarks1 = face_recognition.face_landmarks(image1)
if face_landmarks1:
    landmarks2 = face_landmarks1[0]
    print("Face landmarks:", landmarks2)
else:
    print("No face detected. on image 1")
face_encoding1 = face_recognition.face_encodings(image1, face_landmarks1)[0]

face_landmarks2 = face_recognition.face_landmarks(image2)
face_encoding2 = face_recognition.face_encodings(image2, face_landmarks2)[0]

# Compare face encodings
distance = face_recognition.face_distance([face_encoding1], face_encoding2)[0]

# Create an image with drawings to visualize the faces and landmarks
pil_image = Image.fromarray(image2)
draw = ImageDraw.Draw(pil_image)

for landmark in face_landmarks2[0].values():
    draw.point(landmark, fill=(0, 255, 0))

pil_image.show()

# Print the distance (similarity) between the faces
print("Face distance:", distance)
