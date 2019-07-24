from PIL import Image
import cv2
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from evaluate_new import get_classifier_predictions_new
import numpy as np

import os
import tensorflow as tf
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

IMAGE_SIZE = 128
emotion_list = [
    'Neutral',
    'Happy',
    'Sad',
    'Surprised',
    'Afraid',
    'Disgusted',
    'Angry',
    'Contemptuous'
]

def get_face(pic, count):
    face = []
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        'C://softwares//Anaconda//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (200, 200), (1000, 1000))
    # print(faces)
    faces = np.array(faces)

    if len(faces) > 0:
        cv2.imwrite("frame%d.jpg" % count, pic)  # save frame as JPEG file
        im = Image.open("frame%d.jpg" % count)
        for (x, y, w, h) in faces:
            # look for the biggest one
            area = []
            area.append(w*h)

        ind = np.argmax(area)
        # print('ind:',ind)

        # print("Faces shape = ", faces.shape)
        real_face = faces[ind]
        [x, y, w, h] = real_face
        region = im.crop((x, y, x + w, y + h))
        # print(region)
        region_new = region.resize((IMAGE_SIZE, IMAGE_SIZE))
        face = np.asarray(region_new)
        # print(img.shape)
        region_new.save("./data_video/frame%d.jpg" % count)
        return face
    else:
        return face

def play_anime(emotion_seq, out_win):
    success = True
    # get the video
    n2h = cv2.VideoCapture('affect_video/N2H.avi')
    h2n = cv2.VideoCapture('affect_video/H2N.avi')
    n2a = cv2.VideoCapture('affect_video/N2A.avi')
    a2n = cv2.VideoCapture('affect_video/A2N.avi')
    n2d = cv2.VideoCapture('affect_video/N2D.avi')
    d2n = cv2.VideoCapture('affect_video/D2N.avi')
    n2c = cv2.VideoCapture('affect_video/N2C.avi')
    c2n = cv2.VideoCapture('affect_video/C2N.avi')
    n2f = cv2.VideoCapture('affect_video/N2F.avi')
    f2n = cv2.VideoCapture('affect_video/F2N.avi')
    n2s = cv2.VideoCapture('affect_video/N2S.avi')
    s2n = cv2.VideoCapture('affect_video/S2N.avi')
    n2p = cv2.VideoCapture('affect_video/N2P.avi')
    p2n = cv2.VideoCapture('affect_video/P2N.avi')

    # get the image
    n = cv2.imread('affect_video/neutral_warped.jpg')
    a = cv2.imread('affect_video/angry_warped.jpg')
    h = cv2.imread('affect_video/happy_warped.jpg')
    s = cv2.imread('affect_video/sad_warped.jpg')
    d = cv2.imread('affect_video/disgust_warped.jpg')
    p = cv2.imread('affect_video/surprise_warped.jpg')
    c = cv2.imread('affect_video/contempt_warped.jpg')
    f = cv2.imread('affect_video/fear_warped.jpg')

    # generate list
    video_list = [[n2h, h2n], [n2s, s2n], [n2p, p2n], [n2f, f2n], [n2d, d2n], [n2a, a2n], [n2c, c2n]]
    # print(video_list[1][1])
    image_list = [n, h, s, p, f, d, a, c]

    print(emotion_seq)

    if emotion_seq[0] == emotion_seq[1]:
        cv2.imshow(out_win, image_list[emotion_seq[0]])
    else:
        if emotion_seq[0] == 0:
            while success:
                success, image = video_list[emotion_seq[1]-1][0].read()
                if success is False:
                    break
                cv2.imshow(out_win, image)
                cv2.waitKey(30)
        elif emotion_seq[1] == 0:
            while success:
                success, image = video_list[emotion_seq[0]-1][1].read()
                if success is False:
                    break
                cv2.imshow(out_win, image)
                cv2.waitKey(30)
        else:
            while success:
                success, image = video_list[emotion_seq[0]-1][1].read()
                if success is False:
                    break
                cv2.imshow(out_win, image)
                cv2.waitKey(30)

            success = True

            while success:
                success, image = video_list[emotion_seq[1]-1][0].read()
                if success is False:
                    break
                cv2.imshow(out_win, image)
                cv2.waitKey(30)

    # print(emotion_seq)
    # if emotion_seq[0] == emotion_seq[1]:
    #     pass
    #     # while success:
    #     #     success, image = n2l.read()
    #     #     if success == False:
    #     #         break
    #     #     cv2.imshow("Image Title", image)
    #     #     cv2.waitKey(5)
    # elif emotion_seq[1] - emotion_seq[0] > 0:              # neutral to happy
    #     while success:
    #         success, image = n2l.read()
    #         if success == False:
    #             break
    #         cv2.imshow("Image Title", image)
    #         cv2.waitKey(5)
    # elif emotion_seq[1] - emotion_seq[0] < 0:              # happy to neutral
    #     while success:
    #         success, image = a2n.read()
    #         if success == False:
    #             break
    #         cv2.imshow("Image Title", image)
    #         cv2.waitKey(5)
    # else:
    #     pass
    #     # while success:
    #     #     success, image = l2n.read()
    #     #     if success == False:
    #     #         break
    #     #     cv2.imshow("Image Title", image)
    #     #     cv2.waitKey(5)

if __name__ == '__main__':
    count = 0
    emotion_last = 0

    # define window property
    out_win = "output_style_full_screen"
    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # activate cam
    camcap = cv2.VideoCapture(0)

    print(camcap.isOpened())

    # Load model
    c_path = 'M_ALEX/C_T.h5'
    model = load_model(c_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2D})
    while(1):
        # From video to the image
        success, image_raw = camcap.read()
        # print(image_raw.shape)

        # cv2.imshow("Camera_Image", image_raw)
        # Load image
        # path = 'photo_test/16.jpg'

        # image_raw = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        # image_raw = np.array(image_raw)
        # image_raw = cv2.imread(path)
        # print(image_raw)
        face = get_face(image_raw, count)
        # face = face.resize(IMAGE_SIZE, IMAGE_SIZE)
        # print(face.shape)
        if len(face):
            # Get emotion prediction
            print("get face")
            emotion_new = get_classifier_predictions_new(model, face)
            print('Emotion:', emotion_new, emotion_list[emotion_new])
            emotion_seq = [emotion_last, emotion_new]
            # Play anime according to emotion
            play_anime(emotion_seq, out_win)
            # Reload emotion_last
            emotion_last = emotion_new
        else:
            IM = cv2.imread("affect_video/neutral_warped.jpg")
            cv2.imshow(out_win, IM)
        # IM = cv2.imread("affect_video/neutral_warped.jpg")
        # cv2.imshow(out_win, IM)
        cv2.waitKey(100)

