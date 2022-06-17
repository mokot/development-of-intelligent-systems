#!/usr/bin/python3
import face_recognition
import json

base_path = "../faces/face"

people = [
    ("1", "Gargamel"),
    ("2", "Marija"),
    ("3", "Ana"),
    ("4", "Maja"),
    ("5", "Irena"),
    ("6", "Mojca"),
    ("7", "Nina"),
    ("8", "Mateja"),
    ("9", "Natasa"),
    ("10", "Andreja"),
]

out = []

for (path, name) in people:
    image = face_recognition.load_image_file(base_path + path + ".png")

    boxes = face_recognition.face_locations(image)

    encoding = face_recognition.face_encodings(
        image, boxes, num_jitters=5, model="large"
    )
    
    out.append({"name": name, "encoding": encoding[0].tolist()})

with open("../data/encoding_data.json", "w") as outfile:
    json.dump(out, outfile)
