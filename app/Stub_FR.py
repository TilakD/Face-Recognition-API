from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import pickle
import cv2
from align import AlignDlib
import re
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'QBFJOTLD'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/api": {"origins": "*"}})


def convertImage(imgData):
    """
    Decoding an image from base64 into raw representation.

    :param imgData: Image in base64
    :return: Raw representation of image
    """
	imgstr = re.search(b'base64,(.*)',imgData).group(1)
	with open('app/output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

        
#@app.route('/')
#def index():
#	return render_template('index.html')
		
@app.route('/api',methods=['GET', 'POST'])
@cross_origin(origin='localhost',headers=['Content-Type']) 
def predict():
    """
    Extracts from webcam feed and passes through served model and ML models to recognise faces
    :return: Name of the Person
    """
    
    #Load served models
	channel = implementations.insecure_channel(host, int(port))
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
	request_pb2 = predict_pb2.PredictRequest()
	request_pb2.model_spec.name = 'my_model'
	request_pb2.model_spec.signature_name = 'predict' 

    #Extract image from webcam and convrt it to required format
	imgData = request.get_data()
	convertImage(imgData)
	im_orig = cv2.imread('app/output.png', 1)
	im_orig = im_orig[...,::-1]
		
    #Align image
	alignment = AlignDlib('app/landmarks.dat')
	bb = alignment.getAllFaceBoundingBoxes(im_orig)
	im_aligned = []
	n = []
	for i in bb:
		j = alignment.align(96, im_orig, i, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		im_aligned.append(j)   

	for n in im_aligned:
		n = (n / 255.).astype(np.float32)
		new_n = np.expand_dims(n, axis=0)

    #pass the face cropped images through the server DL model
	request_pb2.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(n, shape=[1, 96, 96, 3]))
	result_future = stub.Predict(request_pb2, 10.) 
	embedded_t = tf.make_ndarray(result_future.outputs['scores']) 
    
    #Pass the extracted embeddings into ML models to get the nearest neighbour 
	example_predic = knn.predict(embedded_t)
	example_prob = svc.predict_proba(embedded_t)
	print(example_prob)
    
	if np.any(example_prob > 0.35):
		example_i = encoder.inverse_transform(example_predic)[0]
		print(example_i)
	else:
		print("Not a face from database...")
        
	maximum = np.max(example_prob, axis=1)
	index_of_maximum = np.where(example_prob == maximum)
	cdsid = {0:'P0',  1:'P1',  2:'P2',  3:'P3',  4:'P4',  5:'P5',  6:'P6',  7:'P7',  8:'P8',  9:'P9',  10:'P10',  11:'P11',  12:'P12',  13:'P13',  14:'P14',  15:'P15',  16:'P16',  17:'P17',  18:'P18',  19:'P19', 20:'P20'}
    
	print(cdsid[float(index_of_maximum[1])])
	response = jsonify(cdsid[float(index_of_maximum[1])])
	return response
	


if __name__ == '__main__':
	server = ""  # Provide Server ip where model is served
	host, port = server.split(':')
	
	# Load the model
	svc_model_save_path = "app/svc_model.sav"
	with open(svc_model_save_path, 'rb') as s:
		svc = pickle.load(s)
	knn_model_save_path = "app/knn_model.sav"
	with open(knn_model_save_path, 'rb') as k:
		knn = pickle.load(k)
        
	app.run(host='0.0.0.0', port=8998, debug=True, ssl_context='adhoc')

