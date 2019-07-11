import keras.backend as K
import tensorflow as tf
from model import create_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def, signature_constants
from tensorflow.contrib.session_bundle import exporter


K.set_learning_phase(0)
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')



export_path = 'inference_graph/V11'
builder = saved_model_builder.SavedModelBuilder(export_path)
prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images':tf.saved_model.utils.build_tensor_info(nn4_small2_pretrained.input)},
            outputs={'scores':tf.saved_model.utils.build_tensor_info(nn4_small2_pretrained.output)},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) 

#signature = predict_signature_def(inputs={'images': nn4_small2_pretrained.input},outputs={'scores': nn4_small2_pretrained.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': prediction_signature})
    builder.save(as_text=True)