from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('Api.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features =[int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predictions=model.predict(final_features)
    output = round(predictions[0], 3)
    return render_template('Api.html',predicted_value="Expected revenue on sales due to Advertisements is {} $".format(output))
if __name__== '__main__':
    app.run(debug=True)


