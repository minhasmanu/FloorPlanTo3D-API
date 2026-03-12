import os
import PIL
import numpy
import uuid
import json
from datetime import datetime





from build_3d_model import build_3d_model, get_normalizer
from mrcnn.config import Config

from mrcnn.model import MaskRCNN

from skimage.color import gray2rgb




from io import BytesIO
from numpy import expand_dims
from flask import Flask, jsonify, request, send_file, send_from_directory

from mrcnn.model import mold_image

import tensorflow as tf
import sys





global _model
global _graph
global cfg
ROOT_DIR = os.path.abspath("./")
WEIGHTS_FOLDER = "./weights"
OUTPUTS_FOLDER = os.path.join(ROOT_DIR, "outputs")
UPLOADS_FOLDER = os.path.join(ROOT_DIR, "uploads")

# Create outputs folder if it doesn't exist
if not os.path.exists(OUTPUTS_FOLDER):
	os.makedirs(OUTPUTS_FOLDER)

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOADS_FOLDER):
	os.makedirs(UPLOADS_FOLDER)

# Blender configuration
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
BLENDER_SCRIPT = os.path.join(ROOT_DIR, "blender_model_builder.py")

from flask_cors import CORS

sys.path.append(ROOT_DIR)

MODEL_NAME = "mask_rcnn_hq"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

application=Flask(__name__)
CORS(application)

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "floorPlan_cfg"
	# number of classes (background + door + wall + window)
	NUM_CLASSES = 1 + 3
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.1
	
@application.before_first_request
def load_model():
	global cfg
	global _model
	model_folder_path = os.path.abspath("./") + "/mrcnn"
	weights_path= os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)
	cfg=PredictionConfig()
	print(cfg.IMAGE_RESIZE_MODE)
	print('==============before loading model=========')
	_model = MaskRCNN(mode='inference', model_dir=model_folder_path,config=cfg)
	print('=================after loading model==============')
	_model.load_weights(weights_path, by_name=True)
	global _graph
	_graph = tf.get_default_graph()


def myImageLoader(imageInput):
	image =  numpy.asarray(imageInput)
	
	if image.ndim != 3:
		image = gray2rgb(image)
		if image.shape[-1] == 4:
			image = image[..., :3]

	h,w,c=image.shape 
	return image,w,h

def getClassNames(classIds):
	result=list()
	for classid in classIds:
		data={}
		if classid==1:
			data['name']='wall'
		if classid==2:
			data['name']='window'
		if classid==3:
			data['name']='door'
		result.append(data)	

	return result				
def normalizePoints(bbx,classNames):
	normalizingX=1
	normalizingY=1
	result=list()
	doorCount=0
	index=-1
	doorDifference=0
	for bb in bbx:
		index=index+1
		if(classNames[index]==3):
			doorCount=doorCount+1
			if(abs(bb[3]-bb[1])>abs(bb[2]-bb[0])):
				doorDifference=doorDifference+abs(bb[3]-bb[1])
			else:
				doorDifference=doorDifference+abs(bb[2]-bb[0])


		result.append([bb[0]*normalizingY,bb[1]*normalizingX,bb[2]*normalizingY,bb[3]*normalizingX])
	return result,(doorDifference/doorCount)	
		

def turnSubArraysToJson(objectsArr):
	result=list()
	for obj in objectsArr:
		data={}
		data['x1']=obj[1]
		data['y1']=obj[0]
		data['x2']=obj[3]
		data['y2']=obj[2]
		result.append(data)
	return result



@application.route('/',methods=['POST'])
def prediction():
	global cfg
	
	# Save uploaded image to uploads folder
	uploaded_file = request.files['image']
	image_filename = f"upload_{uuid.uuid4().hex}_{int(datetime.now().timestamp() * 1000)}.jpg"
	image_filepath = os.path.join(UPLOADS_FOLDER, image_filename)
	uploaded_file.save(image_filepath)
	
	# Load image from saved path
	imagefile = PIL.Image.open(image_filepath)
	image,w,h=myImageLoader(imagefile)
	print(w, h, image.shape)
	scaled_image = mold_image(image, cfg)
	sample = expand_dims(scaled_image, 0)

	global _model
	global _graph
	with _graph.as_default():
		r = _model.detect(sample, verbose=0)[0]
	
	#output_data = model_api(imagefile)
	
	data={}
	bbx=r['rois'].tolist()
	temp,averageDoor=normalizePoints(bbx,r['class_ids'])
	temp=turnSubArraysToJson(temp)
	data['points']=temp
	data['classes']=getClassNames(r['class_ids'])
	data['Width']=w
	data['Height']=h
	data['averageDoor']=averageDoor

	scale = 1 / get_normalizer(data)
	def sample(x, y):
		x *= scale
		y *= scale
		r, g, b = image[int(y), int(x)]
		is_white = r > 64 and g > 64 and b > 64
		return is_white

	gltf = build_3d_model(data, sample_image=sample)

	# Generate unique filename
	filename = f"model_{uuid.uuid4().hex}_{int(datetime.now().timestamp() * 1000)}.glb"
	filepath = os.path.join(OUTPUTS_FOLDER, filename)
	
	# Save GLB file
	with open(filepath, 'wb') as f:
		gltf.write_glb(f)
	
	history_file = os.path.join(ROOT_DIR, "history.json")

# read history
	if os.path.exists(history_file):
		with open(history_file, "r") as f:
			history = json.load(f)
	else:
		history = []

	entry = {
		"id": uuid.uuid4().hex,
		"image": image_filename,
		"model": filename,
		"image_url": f"/uploads/{image_filename}",
		"model_url": f"/outputs/{filename}",
		"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	}

	history.append(entry)

	with open(history_file, "w") as f:
		json.dump(history, f, indent=4)

	# return response to frontend
	return jsonify({
		"model": filename,
		"url": f"/outputs/{filename}",
		"image": image_filename,
		"image_url": f"/uploads/{image_filename}"
	})

@application.route('/uploads/<filename>')
def get_uploaded_image(filename):
    return send_from_directory(UPLOADS_FOLDER, filename)


@application.route('/outputs/<filename>', methods=['GET'])
def download_model(filename):
	"""Serve GLB files from the outputs directory."""
	try:
		return send_from_directory(OUTPUTS_FOLDER, filename, mimetype="model/gltf-binary")
	except Exception as e:
		return jsonify({"error": "File not found"}), 404


@application.route('/history', methods=['GET'])
def get_history():

    history_file = os.path.join(ROOT_DIR, "history.json")

    if not os.path.exists(history_file):
        return jsonify([])

    with open(history_file, "r") as f:
        history = json.load(f)

    history.reverse()

    return jsonify(history)
    

@application.route('/delete/<id>', methods=['DELETE'])
def delete_item(id):

    history_file = os.path.join(ROOT_DIR, "history.json")

    with open(history_file, "r") as f:
        history = json.load(f)

    updated = []
    deleted_item = None

    for item in history:
        if item["id"] == id:
            deleted_item = item
        else:
            updated.append(item)

    if deleted_item:

        image_path = os.path.join(UPLOADS_FOLDER, deleted_item["image"])
        model_path = os.path.join(OUTPUTS_FOLDER, deleted_item["model"])

        if os.path.exists(image_path):
            os.remove(image_path)

        if os.path.exists(model_path):
            os.remove(model_path)

    with open(history_file, "w") as f:
        json.dump(updated, f, indent=4)

    return jsonify({"message": "Deleted"})



if __name__ =='__main__':
	application.debug=True
	print('===========before running==========')
	application.run(host="0.0.0.0", port=8081)
	print('===========after running==========')
