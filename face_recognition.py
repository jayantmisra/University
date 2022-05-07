# Basic Facial Recognition using Raspberry Pi
import face_recognition
import picamera
import numpy as np

camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)

# This part of the code can be edited to add more faces.
# Load a sample picture and learn how to recognize it.
print("Loading known face image(s)")
image1 = face_recognition.load_image_file("")
image1_face_encoding = face_recognition.face_encodings(image1)[0]
image2 = face_recognition.load_image_file("")
image2_face_encoding = face_recognition.face_encodings(image2)[0]
loaded_face_encodings = [image1_face_encoding, image2_face_encoding]
loaded_face_names = ["Person-1", "Person-2"]
##########################################################################################

# Initialize some variables
face_locations = []
face_encodings = []

while True:
    print("Capturing image.")
    # Grab a single frame of video from the RPi camera as a numpy array
    camera.capture(output, format="rgb")

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)

    # Loop over each face found in the frame to see if it's someone we know.
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)

        matches = face_recognition.compare_faces(loaded_face_encodings, face_encoding)
        name = "<Unknown Person>"

        face_distances = face_recognition.face_distance(
            loaded_face_encodings, face_encoding
        )
        bmi = np.argmin(face_distances)

        if matches[bmi]:
            name = loaded_face_names[bmi]

        print("I see someone named {}!".format(name))