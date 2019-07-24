#! /usr/bin/env python

import csv
import os
import cv2
import sys
import numpy as np


def process_images(p, limit=-1):
    with open(p, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in r:
            if 'sub' in row[0]:
                continue
            try:
                valence = float(row[-2])
                arousal = float(row[-1])
                emotion = int(row[-3])
                if (arousal == -2 or valence == -2) and emotion > 7:
                    continue
                x, y, w, h = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]  # x, y, w, h
                path = 'data/' + row[0]
                o_path = 'data_p/' + row[0]
                if not os.path.exists(o_path):
                    image = cv2.imread(path)[int(y):int(y + h), int(x):int(x + w)]
                    image = cv2.resize(image, (256, 256))
                    o_dir = '/'.join(o_path.split('/')[:-1])
                    if not os.path.exists(o_dir):
                        os.makedirs(o_dir)
                    cv2.imwrite(o_path, image)
                count += 1
                print('Processed:', count, end='\r')
                if limit > 0 and count == limit:  # TEMP
                    break
            except Exception:
                continue
    print('Processed:', count)


def process_labels(p):
    labels = []
    paths = []
    count = 0
    with open(p, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in r:
            if 'sub' in row[0]:
                continue
            valence = float(row[-2])
            arousal = float(row[-1])
            emotion = int(row[-3])
            path = 'data_p/' + row[0]
            if emotion > 7 and (arousal == -2 or valence == -2):
                continue
            if not os.path.exists(path):
                print('error: no image')
                continue
            labels.append((emotion, valence, arousal))
            paths.append(path)
            count += 1
            print('Loaded:', count, end='\r')
    print('Loaded:', count)
    return paths, labels


if __name__ == '__main__':
    if '-i' not in sys.argv and '-l' not in sys.argv:
        print('Usage:')
        print('process.py')
        print('           -i    process images from AffectNet training.csv + validation.csv')
        print('                 requires raw images in data dir and outputs to data_p')
        print('           -l    process labels from AffectNet training.csv + validation.csv')
        print('                 outputs to [training/validation]_[paths/labels].npy')
    if '-i' in sys.argv:
        process_images('training.csv')
        process_images('validation.csv')
    if '-l' in sys.argv:
        tp, tl = process_labels('training.csv')
        np.save('training_paths', tp)
        np.save('training_lables', tl)
        vp, vl = process_labels('validation.csv')
        np.save('validation_paths', vp)
        np.save('validation_labels', vl)
