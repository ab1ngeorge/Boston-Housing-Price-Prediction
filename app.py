from flask import Flask, render_template, request, flash
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

# Load the model
try:
    with open('house_price_prediction.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash("Model not loaded. Please try again later.", "error")
        return render_template('index.html')
    
    try:
        # Get all features from the form
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)
        output = round(prediction[0], 2)
        
        return render_template('index.html', 
                             prediction_text=f"Predicted Price: ${output * 1000:,.2f}")
    
    except ValueError:
        flash("Please enter valid numbers for all fields.", "error")
        return render_template('index.html')
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
