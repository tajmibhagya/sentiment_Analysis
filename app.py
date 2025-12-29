from flask import Flask,render_template,request,redirect 
from helper import preprocessing, vectorizer, get_prediction
app= Flask(__name__)

data = dict()
reviews = ['Good Product', 'Bad Product' , 'I like it']
positive = 2 #to display.just for now  
negative = 1

@app.route("/")
def index():
    data['reviews'] = reviews     #add values to data dictionary
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html',data=data)
    #pass data into html

@app.route("/",methods=['POST']) #this runs when we posted comment
def my_post():
    text=request.form['text'] #we take values of comment
    preprocessed_txt = preprocessing(request.form['text'])
    vectorized_txt = vectorizer(preprocessed_txt)
    prediction = get_prediction(vectorized_txt)
    
    if(prediction == 'Negative'):
        global negative
        negative += 1
    else:
        global positive
        positive += 1
        
    reviews.insert(0,text)
    return redirect(request.url)









if __name__ == "__main__":
    app.run()