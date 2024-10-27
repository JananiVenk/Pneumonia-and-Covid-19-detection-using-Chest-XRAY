import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np

def getPrediction(filename):
    model1 = load_model('chest_xray.h5')
    image = load_img('static/images/'+filename, target_size=(256,256))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,256,256,3)
    pred = np.argmax(model1.predict(img)).astype('int32')
    label=["COVID","NORMAL","PNEUMONIA"]
    return label[pred]

