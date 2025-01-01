#use coding:utf-8
import os
import dlib
from skimage import io
import numpy as np
import cv2
import time as tm
from ai_tools import microsoft_demo as msd

class LDM:
    def __init__(self):
        self.predictor_path = "landmarks_68.dat"
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        self.predictor_path = self.get_model(url, self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        
        self.face_rec_model_path = 'face_rec.dat'
        url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
        self.face_rec_model_path = self.get_model(url, self.face_rec_model_path)
        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)
        self.CF = msd.init()

    def imread(self, imgpath):
        return io.imread(imgpath)

    def get_part_landmarks(self, shape, start_index, end_index):
        jaw = []
        for i in range(start_index, end_index):
            jaw.append(np.array((shape.part(i).x, shape.part(i).y)))
        return jaw

    def landmark_list(self, img):
        dets = self.detector(img, 1)
        ldl = []
        facel = []
        for k, d in enumerate(dets):
            shape = self.predictor(img, d)
            ld = {
                'jaw': self.get_part_landmarks(shape, 0, 17),
                'right_brow': self.get_part_landmarks(shape, 17, 22),
                'left_brow': self.get_part_landmarks(shape, 22, 27),
                'nose': self.get_part_landmarks(shape, 27, 36),
                'right_eye': self.get_part_landmarks(shape, 36, 42),
                'left_eye': self.get_part_landmarks(shape, 42, 48),
                'mouth': self.get_part_landmarks(shape, 48, 59),
                'mouth2': self.get_part_landmarks(shape, 60, 67),
                'all': self.get_part_landmarks(shape, 0, 67),
                'shape': shape
            }
            ldl.append(ld)
            facel.append(d)
        return ldl, facel

    def get_model(self, url, predictor_path):
        if not os.path.exists(predictor_path):
            if not os.path.exists(f"{predictor_path}.bz2"):
                os.system(f'wget -O {predictor_path}.bz2 {url}')
            os.system(f'bunzip2 {predictor_path}.bz2')
        return predictor_path

    def landmarks(self, img):
        ldl, facel = self.landmark_list(img)
        helptxt = 'dict[0]_item:jaw,right_brow,left_brow,nose,right_eye,left_eye,mouth,mouth2'
        return ldl, facel, helptxt + ',model_path=' + self.predictor_path

    def face_area_rate(self, img, facel):
        ratel = []
        for face in facel:
            rate = float(face.width()) * float(face.height()) / (img.shape[0] * img.shape[1])
            ratel.append(rate)
        return ratel

    def face_number(self, img, facel):
        return len(facel)

    def draw_rect(self, img, rect):
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

    def face_center_degree(self, img, ldl, facel):
        xdl, ydl, mdl = [], [], []
        for face in facel:
            xd = (face.left() + face.right()) / 2
            xd = abs(xd - img.shape[0] / 2) / (img.shape[0] / 2)
            xd = 1 - xd

            yd = (face.top() + face.bottom()) / 2
            yd = abs(yd - img.shape[1] / 2) / (img.shape[1] / 2)
            yd = 1 - yd

            md = (xd + yd) * 0.5
            xdl.append(xd)
            ydl.append(yd)
            mdl.append(md)
        return xdl, ydl, mdl

    def face_feature(self, img, facel, fc=1):
        ffl = []
        for face in facel:
            shape = self.predictor(img, face)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape, fc)
            ffl.append(face_descriptor)
        return ffl

    def normalized_sigmoid_fkt(self, a, b, x):
        s = 2 / (1 + np.exp(b * (x - a)))
        return s

    def face_compare(self, feature1, feature2, dist_type='cosine'):
        vec1, vec2 = np.array(feature1), np.array(feature2)
        if dist_type == 'euclidean':
            dist = np.linalg.norm(vec1 - vec2)
        elif dist_type == 'manhattan':
            dist = np.linalg.norm(vec1 - vec2, ord=1)
        elif dist_type == 'chebyshev':
            dist = np.linalg.norm(vec1 - vec2, ord=np.inf)
        elif dist_type == 'cosine':
            dist = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return self.normalized_sigmoid_fkt(0, 1.7, dist), dist

    def compare_ffl(self, ff1l, ff2l):
        scorel, index1l, index2l = [], [], []
        for ic1, ff1 in enumerate(ff1l):
            for ic2, ff2 in enumerate(ff2l):
                score, _ = self.face_compare(ff1, ff2, 'euclidean')
                scorel.append(score)
                index1l.append(ic1)
                index2l.append(ic2)
        return scorel, index1l, index2l

    def face_rec(self, img1, img2, threshold=0.8):
        ld1l, face1l, _ = self.landmarks(img1)
        ld2l, face2l, _ = self.landmarks(img2)
        ff1l = self.face_feature(img1, face1l)
        ff2l = self.face_feature(img2, face2l)
        scorel, index1l, index2l = self.compare_ffl(ff1l, ff2l)
        for i in range(len(scorel) - 1, -1, -1):
            if scorel[i] < threshold:
                del scorel[i]
                del index1l[i]
                del index2l[i]
        return {'scorel': scorel, 'face1l': face1l, 'face2l': face2l, 'index1l': index1l, 'incex2l': index2l}

    def face_rec_ms(self, img1, img2, threshold=0.8):
        cv2.imwrite("tmp1.png", img1)
        cv2.imwrite("tmp2.png", img2)
        scorel = msd.ms_face_verify(self.CF, "tmp1.png", "tmp2.png")
        return {'scorel': scorel, 'face1l': [], 'face2l': [], 'index1l': [], 'incex2l': []}

    def has_same_person(self, img1, img2, threshold=0.8, savename='tmp.jpg'):
        ldl1, face1l, _ = self.landmarks(img1)
        ldl2, face2l, _ = self.landmarks(img2)
        ff1l = self.face_feature(img1, face1l)
        ff2l = self.face_feature(img2, face2l)
        scorel, index1l, index2l = self.compare_ffl(ff1l, ff2l)
        sarray = np.array(scorel)
        max_score = sarray.max() if len(sarray) > 0 else 0
        sarray = sarray > threshold
        for face in face1l:
            img1 = self.draw_rect(img1, face)
        for face in face2l:
            img2 = self.draw_rect(img2, face)
        img_a = np.hstack((img1, img2))
        cv2.putText(img_a, f'similarity:{max_score:.2f}', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        cv2.imwrite(savename, img_a)
        return np.sum(sarray), max_score, np.array((len(face1l), len(face2l)))

    def compare_and_score_ffl(self, ff1l, ff2l, threshold=0.8):
        scorel, index1l, index2l = self.compare_ffl(ff1l, ff2l)
        sarray = np.array(scorel)
        max_score = sarray.max() if len(sarray) > 0 else 0
        sarray_t = sarray > threshold
        return np.sum(sarray_t), max_score, sarray

    def compare2dir(self, imagedir1, imagedir2, max_compare_num=30, score_threshold=0.8):
        timecost = 0
        cc1 = cc2 = compare_num = sameper_num = score_ave = 0.0
        max_see_images = 50
        see_images1 = see_images2 = 0
        t1 = tm.time()
        for imf1 in os.listdir(str(imagedir1)):
            img1 = cv2.imread(os.path.join(imagedir1, imf1))
            cc1 += 1
            if compare_num > max_compare_num or see_images1 > max_see_images or see_images2 > max_see_images:
                break
            for imf2 in os.listdir(imagedir2):
                img2 = cv2.imread(os.path.join(imagedir2, imf2))
                rd = self.face_rec(img1, img2, 0.0)
                if compare_num > max_compare_num or see_images1 > max_see_images or see_images2 > max_see_images:
                    break
                if len(rd['face1l']) < 1 or cc1 > 3 or cc2 > 3:
                    see_images1 += cc1
                    see_images2 += cc2
                    cc1 = cc2 = 0
                    break
                if len(rd['face2l']) < 1:
                    see_images2 += 1
                    cc2 = 0
                    continue
                for score in rd['scorel']:
                    score_ave += score
                    compare_num += 1
                    cc2 += 1
                    if score > score_threshold:
                        sameper_num += 1
                        break
                if compare_num > max_compare_num:
                    break
                cc2 += 1
        score_ave /= (compare_num + 1e-7)
        t2 = tm.time()
        return sameper_num / (compare_num + 1e-7), score_ave, compare_num, imagedir1.split('/')[-1], imagedir2.split('/')[-1], t2 - t1, "http://clt.management.vipkid.com.cn/operation/classroom/classroom/"

    def roc(self, mscore, outlier=-2):
        pos, neg = 0, 0
        db = []
        for i in range(mscore.shape[0]):
            for j in range(mscore.shape[1]):
                tmp0, tmp1 = (1, 0) if i == j else (0, 1)
                if mscore[i][j] != outlier:
                    pos += tmp0
                    neg += tmp1
                    db.append([mscore[i][j], tmp0, tmp1])
        db = sorted(db, key=lambda x: x[0], reverse=True)
        
        xy_arr, xy_arr_key, error_equl_pos = [], [], []
        tp, fp = 0., 0.
        keypoint = [0.2, 0.1, 0.01, 0.001]
        i = 0
        for ic, (x, y, score, sumxy) in enumerate(db):
            tp += y
            fp += x
            xy_arr.append([fp / neg, tp / pos, score, fp / neg + tp / pos])
            if ic > 0 and db[ic - 1][1] < 1 - keypoint[i] and y >= 1 - keypoint[i]:
                xy_arr_key.append([x, y, score])
                if i < len(keypoint) - 1:
                    i += 1
            if ic > 0 and db[ic - 1][3] < 1 and sumxy >= 1:
                error_equl_pos.append([x, y, score, sumxy])
        auc = sum((x - prev_x) * y for prev_x, (x, y, _, _) in zip([0] + [v[0] for v in xy_arr[:-1]], xy_arr))
        print(f"the auc is {auc}.")
        print('roc data is :', xy_arr_key)
        print('error_equl_pos:', error_equl_pos)
        return xy_arr, xy_arr_key, error_equl_pos, auc




























        

 
