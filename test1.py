import dlib
import face_recognition
from PIL import Image



# Load the image
image_path1 = r"C:\Users\Hp\Downloads\passport1.jpg"
image_path2 = r"C:\Users\Hp\Downloads\pan card.jpg"

image1 = face_recognition.load_image_file(image_path1)
image2 = face_recognition.load_image_file(image_path2)

# Find face locations in both images
face_locations1 = face_recognition.face_locations(image1)
print(face_locations1)
top, right, bottom, left = face_locations1[0]
extracted_face = image1[top:bottom, left:right]




face_locations2 = face_recognition.face_locations(image2)

top2, right2, bottom2, left2 = face_locations2[0]
extracted_face2 = image2[top2:bottom2, left2:right2]

# Extract face encodings
face_encodings1 = face_recognition.face_encodings(image1)[0]
face_encodings2 = face_recognition.face_encodings(image2)[0]

# Compare the face encodings   

results = face_recognition.compare_faces([face_encodings1],face_encodings2)

print(results)

# Display the images
pil_image1 = Image.fromarray(extracted_face)
pil_image1.show()

pil_image2 = Image.fromarray(extracted_face2)
pil_image2.show()
