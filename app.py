from flask import Flask, render_template, request
import pickle
import numpy as np

model=pickle.load(open('grid_model','rb'))
vector=pickle.load(open('scaling_model','rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    
    if data1=='Male':
        arr=np.array([[np.float(data2),np.float(data3),0.0,1.0]])
    elif data1=='Female':
        arr=np.array([[np.float(data2),np.float(data3),1.0,0.0]])

    
    predict=model.predict(vector.transform((arr)))
    
    return render_template('after.html',data=predict[0])


if __name__ == "__main__":
    app.run(debug=True)