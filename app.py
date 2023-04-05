from flask import Flask,request,render_template
from PIL import Image
import base64
import io
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import load_img ,img_to_array      
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
import pickle
from englisttohindi.englisttohindi import EngtoHindi
# from flask_pymongo import PyMongo


app=Flask(__name__)


model_disease=load_model('AI_models\\disease_detection.h5')
model_insect=load_model('AI_models\\insect_detection.h5')
model_mustard=load_model('AI_models\\mustered.h5')
model_soil=pickle.load(open('AI_models\\crop.pickel','rb'))
model_fertile=pickle.load(open('AI_models\\fertile.pickel','rb'))

app.secret_key='SimranKhan'
# app.config['MONGO_URI']='mongodb+srv://shanuraghuwanshi873:Mongo873873@agrodatabase.krqdjf3.mongodb.net/agrodatabase?retryWrites=true&w=majority'
# mongo=PyMongo(app)




@app.route("/")

def index():
    return render_template('index.html')

@app.route('/index_hindi')
def index_hindi():
    return render_template('index_hindi.html')



@app.route('/disease')
def disease():
    return render_template('disease.html')

des={0: 'Apple___Apple_scab',1:'Apple___Black_rot',2: 'Apple___Cedar_apple_rust',3: 'Apple___healthy',4: 'Blueberry___healthy',5: 'Cherry_(including_sour)___Powdery_mildew',6: 'Cherry_(including_sour)___healthy',7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',8: 'Corn_(maize)___Common_rust_',9: 'Corn_(maize)___Northern_Leaf_Blight',10: 'Corn_(maize)___healthy',11: 'Grape___Black_rot',12: 'Grape___Esca_(Black_Measles)',13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',14: 'Grape___healthy',15: 'Orange___Haunglongbing_(Citrus_greening)',16: 'Peach___Bacterial_spot',17: 'Peach___healthy',18: 'Pepper,_bell___Bacterial_spot',19: 'Pepper,_bell___healthy',20: 'Potato___Early_blight',21: 'Potato___Late_blight',22: 'Potato___healthy',23: 'Raspberry___healthy',24: 'Soybean___healthy',25: 'Squash___Powdery_mildew',26: 'Strawberry___Leaf_scorch',27: 'Strawberry___healthy',28: 'Tomato___Bacterial_spot',29: 'Tomato___Early_blight',30: 'Tomato___Late_blight',31: 'Tomato___Leaf_Mold',32: 'Tomato___Septoria_leaf_spot',33: 'Tomato___Spider_mites Two-spotted_spider_mite',34: 'Tomato___Target_Spot',35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',36: 'Tomato___Tomato_mosaic_virus',37: 'Tomato___healthy'}
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png','jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def model_predict(img_path,model_disease):
    img=load_img(img_path,target_size=(200,200))
    x=img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x,mode='caffe')
    preds=model_disease.predict(x)
    return preds
filename=''          
@app.route('/detect',methods=['GET','POST'])
def detect():
        if request.method=='POST':
            global filename
            f=request.files['file']
            basepath=os.path.dirname(__file__)
            
            filename=secure_filename(f.filename)
            file_path=os.path.join(basepath,'static/upload',filename)
            f.save(file_path)
            preds=model_predict(file_path,model_disease)
            result=np.argmax(preds)

            img=Image.open(file_path)
            data=io.BytesIO()
            img.save(data,'JPEG')
            encode_img_data=base64.b64encode(data.getvalue())
            
            return render_template('disease.html',filename=encode_img_data.decode('UTF-8'),prediction_crop= des[result])
        else:
            return render_template('disease.html')
        
@app.route('/disease_hindi')
def disease_hindi():
    return render_template('disease_hindi.html')

@app.route('/detect_hindi',methods=['GET','POST'])
def detect_hindi():
        if request.method=='POST':
            global filename
            f=request.files['file']
            basepath=os.path.dirname(__file__)
            
            filename=secure_filename(f.filename)
            file_path=os.path.join(basepath,'static/upload',filename)
            f.save(file_path)
            preds=model_predict(file_path,model_disease)
            result=np.argmax(preds)

            img=Image.open(file_path)
            data=io.BytesIO()
            img.save(data,'JPEG')
            encode_img_data=base64.b64encode(data.getvalue())
            msg=des[result]
            res = EngtoHindi(msg)
            output=res.convert
            
            return render_template('disease_hindi.html',filename=encode_img_data.decode('UTF-8'),prediction_crop= output)
        else:
            return render_template('disease_hindi.html')
        

        


@app.route('/soil')
def soil():
    return render_template('soil.html')
@app.route('/predict_soil',methods=['POST','GET'])
def predict_soil():
    if request.method=='POST':  
        features=[int(x) for x in request.form.values()]
        final_features=[np.array(features)]
        output=model_soil.predict(final_features)
        return render_template('soil.html',prediction_crop=f"Crop should be grown in this type of soil is  {output[0].upper()} ")
    else:
        return render_template('soil.html')
    
@app.route('/soil_hindi')
def soil_hindi():
    return render_template('soil_hindi.html')
@app.route('/predict_soil_hindi',methods=['POST','GET'])
def predict_soil_hindi():
    if request.method=='POST':  
        features=[int(x) for x in request.form.values()]
        final_features=[np.array(features)]
        output=model_soil.predict(final_features)
        msg=f"Crop should be grown in this type of soil is  {output[0]} "
        res = EngtoHindi(msg)
        pred=res.convert
        return render_template('soil_hindi.html',prediction_crop=pred)
    else:
        return render_template('soil_hindi.html')



@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')
stype_dict={'Black':0,'Clay':1,'Loamy':2,'Red':3,'Sandy':4}
ctype_dict={'Barley':0,'Cotton':1,'Ground Nuts':2,'Maize':3,'Millets':4,'Oil Seeds':5,'Paddy':6,'Pulses':7,'Sugarcane':8,'Tobacco':9,'Wheat':10}

@app.route('/predict_fertilizer',methods=['POST','GET'])
def predict_fertilizer():
    if request.method=='POST': 
        features=[]

        features.append(float(request.form.get('Temparature')))
        features.append(float(request.form.get('Humidity')))
        features.append(float(request.form.get('Moisture')))
        features.append(int(stype_dict.get(request.form.get('Soil Type'))))
        features.append(int(ctype_dict.get(request.form.get('Crop Type'))))
        features.append(float(request.form.get('Nitrogen')))
        features.append(float(request.form.get('Potassium')))
        features.append(float(request.form.get('Phosphorous')))

        final_features=[np.array(features)]
        output=model_fertile.predict(final_features)

        out_dict={0:'10-26-26',1:'14-35-14',2:'17-17-17',3:'20-20-20',4:'28-28-0',5:'DAP',6:'Urea'}

        return render_template('fertilizer.html',prediction_fertilizer=f"Fertilizer should be used in this type of soil is  {out_dict.get(output[0])} fertilizer ")

@app.route('/fertilizer_hindi')
def fertilizer_hindi():
    return render_template('fertilizer_hindi.html')

@app.route('/predict_fertilizer_hindi',methods=['POST','GET'])
def predict_fertilizer_hindi():
    if request.method=='POST': 
        features=[]

        features.append(float(request.form.get('Temparature')))
        features.append(float(request.form.get('Humidity')))
        features.append(float(request.form.get('Moisture')))
        features.append(int(stype_dict.get(request.form.get('Soil Type'))))
        features.append(int(ctype_dict.get(request.form.get('Crop Type'))))
        features.append(float(request.form.get('Nitrogen')))
        features.append(float(request.form.get('Potassium')))
        features.append(float(request.form.get('Phosphorous')))

        final_features=[np.array(features)]
        output=model_fertile.predict(final_features)

        out_dict={0:'10-26-26',1:'14-35-14',2:'17-17-17',3:'20-20-20',4:'28-28-0',5:'DAP',6:'Urea'}

        msg=f"in this type of soil {out_dict.get(output[0])} should be use as a fertilizer "
        res = EngtoHindi(msg)
        pred=res.convert

        return render_template('fertilizer_hindi.html',prediction_fertilizer=pred)
    

@app.route('/about')
def about():
    return render_template('about_us.html')

        
@app.route('/contact' ,methods=['GET','POST'])
def contact():
    if request.method=='POST':
        name=request.form.get('name')
        email=request.form.get('email')
        message=request.form.get('message')
        # mongo.db.agrodatabase.insert_one({'name': name, 'email': email, 'message': message})
    return render_template('contact.html')

if __name__=="__main__":
    app.run(debug=True)