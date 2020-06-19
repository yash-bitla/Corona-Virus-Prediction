from flask import Flask, render_template, request
import json

app = Flask(__name__)
@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      Date = result["Date"]
      Country = result["Country"]
      if(Country == "In"):
         Prediction = json.load(open("India_pred.txt"))
      else:
         txt_file = Country
         txt_file = txt_file + "_pred" ".txt"
         Prediction = json.load(open(txt_file))
      Number = Prediction[Date]
      return render_template("result.html",result = Prediction, Country = Country, Date = Date, Number = Number  )

if __name__ == '__main__':
   app.run(debug = True)