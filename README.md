# Face_Recognition
Face Recognition flask app for webcam feed with tensorflow served triplet loss based DL model.

Further details in [here](https://github.com/TilakD/Face-Recognition-API/blob/master/Face%20Recognition%20app.pdf)

## Tensorflow serving commands
```
tensorflow_model_server --port=9001 --model_name=frlab --model_base_path=/home/dtilak/fr


docker run -d -p 9000:8500 -p 9001:8501 --mount type=bind,source=/home/dtilak/fr, target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving 
```


## Flask Command
`docker run -it -p 8998:8998 face_recognition/fr_flask_app:V1 bash`
