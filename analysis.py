import cv2, numpy, dlib
from imutils import face_utils
from dotenv import load_dotenv
import os, sys

class ImageCalibration:
    CALIBRATION_FILENAME = "calibration_matrix.npz"

    def __init__(self, dir_path:str, cal_params:tuple)->None:
        self.__dir_path = dir_path
        self.__rows, self.__cols, self.__size = cal_params
        self.__cal_filepath = os.path.join(self.__dir_path, self.CALIBRATION_FILENAME)
        self.__mtx, self.__dist = None, None
        if os.path.exists(self.__cal_filepath):
            cal_mtx = numpy.load(self.__cal_filepath)
            self.__mtx, self.__dist = cal_mtx["camera_matrix"], cal_mtx["distortion_coefficients"]
        else:
            mtx, dist = self._cal_camera()
            if mtx is None or dist is None:
                print(">> Calculate camera matrix Failed. Stop this program...")
                sys.exit()
            else:
                self.__mtx, self.__dist = mtx, dist

    def _cal_camera(self)->tuple:
        corners_list = []
        for filename in os.listdir(self.__dir_path):
            if filename.lower().endswith((".bmp", "jpg", "png")):
                gray_img = cv2.imread(os.path.join(self.__dir_path, filename), cv2.COLOR_BGR2GRAY)
                ret, corners = self._find_circles_grid(gray_img)
                if ret:
                    corners_list.append(corners)
        if len(corners_list) > 0:
            obj_pt = numpy.zeros((self.__rows * self.__cols, 3), numpy.float32)
            obj_pt[:, :2] = numpy.mgrid[0:self.__cols, 0:self.__rows].T.reshape(-1, 2) * self.__size
            _, mtx, dist, r_vecs, t_vecs = cv2.calibrateCamera([obj_pt for _ in range(len(corners_list))], corners_list, (gray_img.shape[1], gray_img.shape[0]), None, None,flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
            numpy.savez(self.__cal_filepath, camera_matrix=mtx, distortion_coefficients=dist, rotation_vectors=r_vecs, translation_vectors=t_vecs)
            return mtx, dist
        else:
            return None, None

    def _find_circles_grid(self, gray_img:numpy.ndarray)->tuple:
        ret, corners = cv2.findCirclesGrid(gray_img, (self.__cols, self.__rows), None)
        if not ret:
            return cv2.findCirclesGrid(cv2.bitwise_not(gray_img), (self.__cols, self.__rows), None)
        return ret, corners

    def undistort_img(self, img:numpy.ndarray)->numpy.ndarray:
        return cv2.undistort(img, self.__mtx, self.__dist, None, self.__mtx)


class FaceDetection:
    PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"

    def __init__(self, dir_path:str)->None:
        # https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
        self.__face_detector = dlib.get_frontal_face_detector()
        self.__face_predictor = dlib.shape_predictor(os.path.join(dir_path, self.PREDICTOR_FILENAME))

    def detect_faces(self, gray_img:numpy.ndarray)->numpy.ndarray:
        return self.__face_detector(gray_img)

    def detect_landmarks(self, img:numpy.ndarray)->list:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = []
        for face in self.detect_faces(gray_img):
            landmarks = face_utils.shape_to_np(self.__face_predictor(gray_img, face))
            result.append({"face":face,"landmark":landmarks})
        return result


class ImageRegistration:
    REGISTRATION_FILENAME = "homography_matrix.npy"

    def __init__(self, reg_params:tuple, dir_path:str)->None:
        self.__rows, self.__cols = reg_params
        self.__homography_mtx_filepath = os.path.join(dir_path, self.REGISTRATION_FILENAME)
        self.__mtx = None 

    def save_homography_mtx(self, inf_img:numpy.ndarray, vis_img:numpy.ndarray)->tuple:
        inf_ret, inf_corners = self._find_circles_grid(inf_img)
        vis_ret, vis_corners = self._find_circles_grid(vis_img)
        if not inf_ret or not vis_ret:
            return None, None
        numpy.save(self.__homography_mtx_filepath, cv2.findHomography(vis_corners, inf_corners, cv2.RANSAC, 5.0)[0])
        self.__mtx = numpy.load(self.__homography_mtx_filepath)
        result_inf_img = cv2.drawChessboardCorners(inf_img, (self.__cols, self.__rows), inf_corners, inf_ret)
        result_vis_img = cv2.drawChessboardCorners(vis_img, (self.__cols, self.__rows), vis_corners, vis_ret)
        return result_inf_img, result_vis_img

    def _find_circles_grid(self, img:numpy.ndarray)->tuple:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(gray_img, (self.__cols, self.__rows), None)
        if not ret:
            return cv2.findCirclesGrid(cv2.bitwise_not(gray_img), (self.__cols, self.__rows), None)
        return ret, corners

    def _manual_find_circles_grid(self, img:numpy.ndarray)->tuple:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        params = [10, 80, 35, 25]
        corners = []
        while True:
            circles = numpy.uint16(numpy.around(cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.5, minDist=params[0], param1=params[1], param2=params[2], minRadius=0, maxRadius=params[3])))
            copied_img = img.copy()
            print(">> If change parameters/choose circles/end manual mode, put 'c'/'f'/'e'.\n-->")
            for i, circle in enumerate(circles[0, :]):
                cv2.circle(copied_img, (circle[0], circle[1]), circle[2], ((i%2)*255, (i%3)*120, (i%5)*50), 2)
                cv2.circle(copied_img, (circle[0], circle[1]), 2, ((i%2)*255, (i%3)*120, (i%5)*50), 3)
                cv2.putText(copied_img, text=f'{i}', org=(circle[0], circle[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=((i%2)*255, (i%3)*120, (i%5)*50), thickness=1, lineType=cv2.LINE_4)
            cv2.imshow("circles", copied_img)
            key = cv2.waitKey() & 0xFF
            if key == ord("f"):
                print(">> Select circles from upper left to lower right.")
                for _ in range(self.__rows * self.__cols):
                    num = int(input("-->"))
                    corners.append(circles[0, :][num][:2])
                corners =numpy.array([corners])
                break
            elif key == ord("c"):
                print(">> Which parameter change?")
                change_type = int(input(">> minDis:'0' / param1:'1' / param2:'2' / maxRadius:'3'\n-->"))
                new_param = input(f"from {params[change_type]} to ? -->")
                params[change_type] = int(new_param) if change_type!=0 else float(new_param)
            elif key == ord("e"):
                return False, None
            else:
                continue
        cv2.destroyAllWindows()
        return True, corners

    def reg_img(self, inf_img:numpy.ndarray, vis_img:numpy.ndarray)->numpy.ndarray:
        if self.__mtx is None:
            return None
        return cv2.warpPerspective(vis_img, self.__mtx, (inf_img.shape[1], inf_img.shape[0]))


class Analyze:
    load_dotenv()
    TEMP_R2 = float(os.environ["TEMP_R2"])
    TEMP_INTERCEPT = float(os.environ["TEMP_INTERCEPT"])
    VOL_R2 = float(os.environ["VOL_R2"])
    VOL_INTERCEPT = float(os.environ["VOL_INTERCEPT"])
    def __init__(self, inf_cal:ImageCalibration, vis_cal:ImageCalibration, face_detec:FaceDetection, img_reg:ImageRegistration):
        self.__inf_img_cal = inf_cal
        self.__vis_img_cal = vis_cal
        self.__face_detec = face_detec
        self.__img_reg = img_reg

    def main(self, inf_movie_filepath:str, vis_movie_filepath:str)->tuple:
        print(">> Analysis.main() START")
        detect_results = []
        data = []
        fps = 0
        inf_capture = cv2.VideoCapture(inf_movie_filepath)
        vis_capture = cv2.VideoCapture(vis_movie_filepath)
        if int(inf_capture.get(cv2.CAP_PROP_FPS)) != int(vis_capture.get(cv2.CAP_PROP_FPS)):
            return None, None
        else:
            fps = int(inf_capture.get(cv2.CAP_PROP_FPS))
        while True:
            inf_ret, inf_frame = inf_capture.read()
            vis_ret, vis_frame = vis_capture.read()
            if not inf_ret or not vis_ret:
                break
            inf_frame = self.__inf_img_cal.undistort_img(inf_frame)
            vis_frame = self.__vis_img_cal.undistort_img(vis_frame)
            detect_results.append(self.__face_detec.detect_landmarks(self.__img_reg.reg_img(inf_frame, vis_frame)))
            if len(detect_results[-1]) == 0:
                if len(detect_results) == 1:
                    data.append(0)
                else:
                    data.append(data[-1])
            else:
                for face in detect_results[-1]:
                    pixs = self._get_pix_in_poly(cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY), face["landmark"][48:60])
                    data.append(self._cal(pixs))
        inf_capture.release()
        vis_capture.release()
        result = self._analysis_flow_vol(data, 30)
        return result, fps

    def _get_pix_in_poly(self, img:numpy.ndarray, poly_pts:list)->list:
        mask = numpy.zeros_like(img, dtype=numpy.uint8)
        cv2.fillPoly(mask, [poly_pts], 255)
        pixcels = img[numpy.where(mask > 0)]
        return pixcels

    def _cal(self, data:list)->float:
        result = numpy.mean(data)
        return result

    def _analysis_flow_vol(self, data:list, win_size:int)->list:
        print(">> Analysis._analysis_flow_vol() START")
        smoothing_data = [sum(data[i : i + win_size]) / win_size for i in range(len(data) - win_size + 1)]
        normalization_data = [(data - min(smoothing_data)) / (max(smoothing_data) - min(smoothing_data)) for data in smoothing_data]
        temp = [data * self.TEMP_R2 + self.TEMP_INTERCEPT for data in normalization_data]
        delta_temp =  [now - prev for prev, now in zip(temp, temp[1:])]
        normalization_delta_temp = [(data - min(delta_temp)) / (max(delta_temp) - min(delta_temp)) for data in delta_temp]
        volume = [data * self.VOL_R2 - self.VOL_INTERCEPT for data in normalization_delta_temp]
        normalization_volume = [(data - min(volume)) / (max(volume) - min(volume)) for data in volume]
        return normalization_volume

    def _geterate_homography_mtx(self, inf_movie_filepath: str, vis_movie_filepath: str):
        gray_inf_img = self._get_img_from_movie(inf_movie_filepath, True)
        gray_vis_img = self._get_img_from_movie(vis_movie_filepath, False)
        result_imgs = self.__img_reg.save_homography_mtx(gray_inf_img, gray_vis_img)
        return result_imgs

    def _get_img_from_movie(self, path:str, is_inf:bool):
        video = cv2.VideoCapture(path)
        ret, frame = video.read()
        while ret:
            ret, return_frame = video.read()
            if ret:
                frame = return_frame

        if is_inf:
            return self.__inf_img_cal.undistort_img(frame)
        else:
            return self.__vis_img_cal.undistort_img(frame)





