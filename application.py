import os
import PIL
import numpy

from build_3d_model import build_3d_model, get_normalizer
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from skimage.color import gray2rgb

from numpy import expand_dims
from flask import Flask, jsonify, request

from mrcnn.model import mold_image

import tensorflow as tf
import sys
from flask import send_from_directory

# ------------------ FOLDERS ------------------

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------

global _model
global _graph
global cfg

ROOT_DIR = os.path.abspath("./")
WEIGHTS_FOLDER = "./weights"

from flask_cors import CORS

sys.path.append(ROOT_DIR)

MODEL_NAME = "mask_rcnn_hq"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

application = Flask(__name__)
cors = CORS(application, resources={r"/*": {"origins": "*"}})

# ------------------ CONFIG ------------------

class PredictionConfig(Config):

    NAME = "floorPlan_cfg"

    NUM_CLASSES = 1 + 3

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.1


# ------------------ LOAD MODEL ------------------

@application.before_first_request
def load_model():

    global cfg
    global _model

    model_folder_path = os.path.abspath("./") + "/mrcnn"

    weights_path = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)

    cfg = PredictionConfig()

    print('==============before loading model=========')

    _model = MaskRCNN(
        mode='inference',
        model_dir=model_folder_path,
        config=cfg
    )

    print('=================after loading model==============')

    _model.load_weights(weights_path, by_name=True)

    global _graph

    _graph = tf.get_default_graph()


# ------------------ IMAGE LOADER ------------------

def myImageLoader(imageInput):

    image = numpy.asarray(imageInput)

    if image.ndim != 3:

        image = gray2rgb(image)

        if image.shape[-1] == 4:

            image = image[..., :3]

    h, w, c = image.shape

    return image, w, h


# ------------------ CLASS NAMES ------------------

def getClassNames(classIds):

    result = []

    for classid in classIds:

        data = {}

        if classid == 1:
            data['name'] = 'wall'

        if classid == 2:
            data['name'] = 'window'

        if classid == 3:
            data['name'] = 'door'

        result.append(data)

    return result


# ------------------ NORMALIZE ------------------

def normalizePoints(bbx, classNames):

    normalizingX = 1
    normalizingY = 1

    result = []

    doorCount = 0

    index = -1

    doorDifference = 0

    for bb in bbx:

        index = index + 1

        if (classNames[index] == 3):

            doorCount = doorCount + 1

            if abs(bb[3] - bb[1]) > abs(bb[2] - bb[0]):

                doorDifference = doorDifference + abs(bb[3] - bb[1])

            else:

                doorDifference = doorDifference + abs(bb[2] - bb[0])

        result.append([
            bb[0] * normalizingY,
            bb[1] * normalizingX,
            bb[2] * normalizingY,
            bb[3] * normalizingX
        ])

    return result, (doorDifference / doorCount)


# ------------------ JSON FORMAT ------------------

def turnSubArraysToJson(objectsArr):

    result = []

    for obj in objectsArr:

        data = {}

        data['x1'] = obj[1]

        data['y1'] = obj[0]

        data['x2'] = obj[3]

        data['y2'] = obj[2]

        result.append(data)

    return result


# ------------------ PREDICTION API ------------------

@application.route('/', methods=['POST'])

def prediction():

    global cfg

    file = request.files['image']

    filename = file.filename

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(image_path)

    imagefile = PIL.Image.open(image_path)

    image, w, h = myImageLoader(imagefile)

    scaled_image = mold_image(image, cfg)

    sample = expand_dims(scaled_image, 0)

    global _model
    global _graph

    with _graph.as_default():

        r = _model.detect(sample, verbose=0)[0]

    data = {}

    bbx = r['rois'].tolist()

    temp, averageDoor = normalizePoints(bbx, r['class_ids'])

    temp = turnSubArraysToJson(temp)

    data['points'] = temp

    data['classes'] = getClassNames(r['class_ids'])

    data['Width'] = w

    data['Height'] = h

    data['averageDoor'] = averageDoor

    scale = 1 / get_normalizer(data)

    def sample(x, y):

        x *= scale

        y *= scale

        r, g, b = image[int(y), int(x)]

        is_white = r > 64 and g > 64 and b > 64

        return is_white

    gltf = build_3d_model(data, sample_image=sample)

    glb_filename = filename.split(".")[0] + ".glb"

    glb_path = os.path.join(OUTPUT_FOLDER, glb_filename)

    with open(glb_path, "wb") as f:

        gltf.write_glb(f)

    return jsonify({
        "model": glb_filename
    })


# ------------------ SERVE MODEL ------------------

@application.route('/outputs/<filename>')

def serve_model(filename):

    return send_from_directory(OUTPUT_FOLDER, filename)


# ------------------ SERVE IMAGE ------------------

@application.route('/uploads/<filename>')

def serve_image(filename):

    return send_from_directory(UPLOAD_FOLDER, filename)


# ------------------ HISTORY ------------------

@application.route('/history', methods=['GET'])

def get_history():

    uploads = os.listdir(UPLOAD_FOLDER)

    outputs = os.listdir(OUTPUT_FOLDER)

    history = []

    for img in uploads:

        name = img.split(".")[0]

        model = name + ".glb"

        if model in outputs:

            history.append({
                "image": img,
                "model": model
            })

    return jsonify(history)


# ------------------ DELETE FILE ------------------

@application.route('/delete/<filename>', methods=['DELETE'])

def delete_file(filename):

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    model_name = filename.split(".")[0] + ".glb"

    model_path = os.path.join(OUTPUT_FOLDER, model_name)

    try:

        if os.path.exists(image_path):
            os.remove(image_path)

        if os.path.exists(model_path):
            os.remove(model_path)

        return jsonify({
            "message": "Deleted successfully"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ------------------ RUN SERVER ------------------

if __name__ == '__main__':

    application.debug = True

    print('===========before running==========')

    application.run(host="0.0.0.0", port=8081)

    print('===========after running==========')