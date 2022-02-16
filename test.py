import time

import matplotlib.pyplot as plt
from facelib import AgeGenderEstimator, FaceDetector

"""
ok: 年龄段  性别  
背包  衣服颜色  戴眼镜
"""


# ============ detect img
img = plt.imread('facelib/imgs/mnls.jpg')
face_detector = FaceDetector()
age_gender_detector = AgeGenderEstimator()
start = time.time()
faces, boxes, scores, landmarks = face_detector.detect_align(img)
genders, ages = age_gender_detector.detect(faces)
fps = 1 / (time.time() - start)  # fps 0.8
print(genders, ages)


# ============ detect webcam
# from facelib import WebcamAgeGenderEstimator
# estimator = WebcamAgeGenderEstimator()
# estimator.run()