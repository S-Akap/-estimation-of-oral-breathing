from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import cv2
import os

from analysis import ImageCalibration, FaceDetection, ImageRegistration, Analyze

FORM_DATA = [
    {"id":"thermal-movie","filename":"thermal_movie","filename_jp":"遠赤外線動画"},
    {"id":"visible-movie","filename":"visible_movie","filename_jp":"可視光動画"},
    {"id":"reg-thermal","filename":"reg_thermal","filename_jp":"レジストレーション用遠赤外線動画"},
    {"id":"reg-visible","filename":"reg_visible","filename_jp":"レジストレーション用可視光動画"},
]

CALIBRATION_PARAMETERS = (7, 6, 1.0)
REGISTRATION_PARAMETERS = (5,4)

UPLOAD_DIRECTORY = os.path.join("uploads")
THERMAL_CALIBRATION_DIRECTORY = os.path.join("inf_calibration")
VISIBLE_CALIBRATION_DIRECTORY = os.path.join("vis_calibration")
PREDICTOR_DIRECTORY = os.path.join("predictor")

ALLOWED_EXTENSIONS = set(["mp4", "avi"])

def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, static_folder = "./static")

app.config["UPLOAD_DIR"] = UPLOAD_DIRECTORY
app.config["INF_CAL_DIR"] = THERMAL_CALIBRATION_DIRECTORY
app.config["VIS_CAL_DIR"] = VISIBLE_CALIBRATION_DIRECTORY
app.config["PREDICTOR_DIR"] = PREDICTOR_DIRECTORY

analyze = Analyze(
    ImageCalibration(app.config["INF_CAL_DIR"], CALIBRATION_PARAMETERS), 
    ImageCalibration(app.config["VIS_CAL_DIR"], CALIBRATION_PARAMETERS), 
    FaceDetection(app.config["PREDICTOR_DIR"]),
    ImageRegistration(REGISTRATION_PARAMETERS, app.config["UPLOAD_DIR"])
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        for data in FORM_DATA:
            filename = data.get("filename")
            if filename not in request.files:
                return render_template("error.html", error_num = "00", msg="ファイルのアップロードに失敗しました。")

            file = request.files[filename]
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config["UPLOAD_DIR"], f"{filename}.{secure_filename(file.filename).rsplit('.', 1)[1]}"))
            else:
                return render_template("error.html", error_num = "01", msg="ファイルのアップロードに失敗しました。")
        return redirect(url_for("result"))
    else:
        return render_template("upload.html", form_data = FORM_DATA)

@app.route("/result")
def result():
    inf_movie_filepath, vis_movie_filepath = None, None
    for filename in os.listdir(os.path.join(app.config["UPLOAD_DIR"])):
        if FORM_DATA[2].get("filename") in filename:
            inf_movie_filepath = os.path.join(app.config["UPLOAD_DIR"], filename)
        if FORM_DATA[3].get("filename") in filename:
            vis_movie_filepath = os.path.join(app.config["UPLOAD_DIR"], filename)
    if inf_movie_filepath is None or vis_movie_filepath is None:
        return render_template("error.html", error_num = "10", msg="ファイルのアップロードに失敗しました。")

    inf_img, vis_img = analyze._geterate_homography_mtx(inf_movie_filepath, vis_movie_filepath)
    if inf_img is None or vis_img is None:
        return render_template("error.html", error_num = "11", msg="2動画間の対応付けに失敗しました。")

    cv2.imwrite(os.path.join(app.config["UPLOAD_DIR"], f"{FORM_DATA[2].get("filename")}.jpg"), inf_img)
    cv2.imwrite(os.path.join(app.config["UPLOAD_DIR"], f"{FORM_DATA[3].get("filename")}.jpg"), vis_img)

    return render_template("result.html")

@app.route("/uploads/<filename>")
def download_file(filename):
    return send_from_directory("uploads", filename, as_attachment=True)

@app.route("/analysis")
def analysis():
    inf_movie_filepath, vis_movie_filepath = None, None
    for filename in os.listdir(os.path.join(app.config["UPLOAD_DIR"])):
        if FORM_DATA[0].get("filename") in filename:
            inf_movie_filepath = os.path.join(app.config["UPLOAD_DIR"], filename)
        if FORM_DATA[1].get("filename") in filename:
            vis_movie_filepath = os.path.join(app.config["UPLOAD_DIR"], filename)
    data, fps = analyze.main(inf_movie_filepath, vis_movie_filepath)

    json_data = {"labels":[i // fps for i in range(len(data))], "data":data}

    return jsonify(json_data)

if __name__ == "__main__":
    app.run()
