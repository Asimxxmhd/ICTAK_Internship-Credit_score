from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#load the model
model = pickle.load(open('credit_score.pkl','rb'))

@app.route('/')
def home():
    result=''
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST','GET'])
def predict():
    occupation = int(request.form['OCCUPATION'])
    creditmix = int(request.form['Credit_Mix'])
    payminamo = int(request.form['Payment_of_Min_Amount'])
    paybeh = int(request.form['Payment_Behaviour'])
    bankacc = int(request.form['Num_Bank_Accounts'])
    creditcar = int(request.form['Num_Credit_Card'])
    intrate = int(request.form['Interest_Rate'])
    noloan = int(request.form['Num_of_Loan'])
    
    # Make prediction and map the result to 'GOOD' or 'STANDARD' or 'BAD'
    result_code = model.predict([[occupation, creditmix, payminamo, paybeh, bankacc, creditcar, intrate, noloan]])[0]
    result = 'GOOD' if result_code == 0 else 'STANDARD' if result_code==1 else 'BAD'
    
    return render_template('index.html', result=result, occupation=occupation, creditmix=creditmix, payminamo=payminamo, paybeh=paybeh, bankacc=bankacc, creditcar=creditcar, intrate=intrate, noloan=noloan)
   

if __name__ == '__main__':
    app.run(debug=True)