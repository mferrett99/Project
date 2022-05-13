import face_recognition
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import glob
import cv2
import os, sys

import cProfile, pstats, io

import numba
from numba import jit
import time




counter = 0

def profile(fnc):
    
    
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        stats = pstats.Stats(pr)
        stats.dump_stats('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/export_data/output.txt')
        
        return retval

    return inner

#@profile
def start(images):
    start = time.time()
    global counter
    charlie_image_1 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_1/person_1_face-1.jpg')
    charlie_image_2 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_1/person_1_face-2.jpg')
    charlie_image_3 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_1/person_1_face-3.jpg')
    charlie_image_4 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_1/person_1_face-4.jpg')
    charlie_image_5 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_1/person_1_face-5.jpg')


    danny_image_1 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_2/person_2_face-1.jpg')
    danny_image_2 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_2/person_2_face-2.jpg')
    danny_image_3 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_2/person_2_face-3.jpg')
    danny_image_4 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_2/person_2_face-4.jpg')
    danny_image_5 = face_recognition.load_image_file('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/known/person_2/person_2_face-5.jpg')


    

    try:
       
        charlie_image_encoding_1 = face_recognition.face_encodings(charlie_image_1)[0]
        charlie_image_encoding_2 = face_recognition.face_encodings(charlie_image_2)[0]
        charlie_image_encoding_3 = face_recognition.face_encodings(charlie_image_3)[0]
        charlie_image_encoding_4 = face_recognition.face_encodings(charlie_image_4)[0]
        charlie_image_encoding_5 = face_recognition.face_encodings(charlie_image_5)[0]

        danny_image_encoding_1 = face_recognition.face_encodings(danny_image_1)[0]
        danny_image_encoding_2 = face_recognition.face_encodings(danny_image_2)[0]
        danny_image_encoding_3 = face_recognition.face_encodings(danny_image_3)[0]
        danny_image_encoding_4 = face_recognition.face_encodings(danny_image_4)[0]
        danny_image_encoding_5 = face_recognition.face_encodings(danny_image_5)[0]


    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()

    known_faces = [
        charlie_image_encoding_1,
        charlie_image_encoding_2,
        charlie_image_encoding_3,
        charlie_image_encoding_4,
        charlie_image_encoding_5,

        danny_image_encoding_1,
        danny_image_encoding_2,
        danny_image_encoding_3,
        danny_image_encoding_4,
        danny_image_encoding_5,
    ]

    known_face_names = [
        "Charlie",
        "Charlie",
        "Charlie",
        "Charlie",
        "Charlie",
        "Danny",
        "Danny",
        "Danny",
        "Danny",
        "Danny",
    ]

    face_locations = face_recognition.face_locations(images)
    face_encodings = face_recognition.face_encodings(images, face_locations)



    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(images)
    draw = ImageDraw.Draw(pil_image)






    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            draw.ellipse(((left, top), (right, bottom)), fill = "blue")

    del draw
    path = 'C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/edited'
    name = ('saved_{}.jpg').format(counter)
    pil_image.save('C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/edited/'+name, 'JPEG', quality = 100)
    counter += 1
    end=time.time()
    print(end - start)
    

    






def resize():
    path = r'C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/unknown/'


    dirs = os.listdir( path )
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            im = im.convert("RGB")
            f, e = os.path.splitext(path+item)
            imResize = im.resize((1920, 1080))
            imResize.save(f+ '.jpg', 'JPEG', quality = 100)


def main():
    
    resize()
    images = [cv2.imread(file) for file in glob.glob("C:/Users/Matt/Desktop/Files/Uni/University/Year 3/Project/New/unknown/*.jpg")]

    i = 0
    while i != len(images):
        image = images[i]
        i += 1
        start(image)




    #start(new)


if __name__ == '__main__':
    main()



#https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py
#https://stackoverflow.com/questions/69963555/face-recognition-with-face-recognition-and-cv2-empty-encoding-error-python
#https://www.geeksforgeeks.org/python-multiple-face-recognition-using-dlib/
#https://stackoverflow.com/questions/21517879/python-pil-resize-all-images-in-a-folder
#https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py
#https://note.nkmk.me/en/python-opencv-bgr-rgb-cvtcolor/
#https://buildmedia.readthedocs.org/media/pdf/face-recognition/latest/face-recognition.pdf