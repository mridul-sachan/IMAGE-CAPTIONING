from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img): 
        img = img.resize((224, 224))  
        img = img.convert('RGB')  
        x = image.img_to_array(img)  
        x = np.expand_dims(x, axis=0)  
        x = preprocess_input(x)  
        feature = self.model.predict(x)[0]  
        return feature / np.linalg.norm(feature) 


'''
# UPLOAD_FOLDER = os.path.join('static','uploads')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello(caption=None, image_path = None):
    return render_template('hello.html', caption=caption)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    
    caption= get_captions(f)
    #caption = "test_caption"
    #return redirect(url_for('hello', caption=caption))
    return render_template('hello.html', caption=caption, image_path=f)
'''

