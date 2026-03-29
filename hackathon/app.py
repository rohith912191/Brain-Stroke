import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
from functools import wraps
import joblib

app = Flask(__name__)
base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(base_dir, 'users.db')
db = SQLAlchemy(app)

try:
    model = joblib.load(os.path.join(base_dir, 'stroke_model.joblib'))
except Exception as e:
    print(f"Model file not found or failed to load: {e}")
    model = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    contact = db.Column(db.String(20), nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)

    def calculate_age(self):
        today = datetime.today()
        return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))

    def calculate_bmi(self):
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid email or password")
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            dob = datetime.strptime(request.form['dob'], '%Y-%m-%d')
            gender = request.form['gender']
            contact = request.form['contact']
            height = float(request.form['height'])
            weight = float(request.form['weight'])

            if User.query.filter_by(email=email).first():
                return render_template('signup.html', error="Email already registered")

            user = User(
                name=name,
                email=email,
                password=generate_password_hash(password),
                dob=dob,
                gender=gender,
                contact=contact,
                height=height,
                weight=weight
            )
            
            db.session.add(user)
            db.session.commit()
            
            return redirect(url_for('login'))
            
        except Exception as e:
            return render_template('signup.html', error=str(e))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def view_profile():
    user = User.query.get(session['user_id'])
    return render_template('profile.html', 
                         user=user,
                         age=user.calculate_age(),
                         bmi=user.calculate_bmi())

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        try:
            user.name = request.form['name']
            user.contact = request.form['contact']
            user.height = float(request.form['height'])
            user.weight = float(request.form['weight'])
            
            db.session.commit()
            return redirect(url_for('view_profile'))
        except Exception as e:
            return render_template('edit_profile.html', user=user, error=str(e))
    
    return render_template('edit_profile.html', user=user)

@app.route('/')
@login_required
def home():
    user = User.query.get(session['user_id'])
    today = datetime.today()
    age = today.year - user.dob.year - ((today.month, today.day) < (user.dob.month, user.dob.day))
    return render_template('stroke.html', 
                         user=user,
                         age=age,
                         default_values={
                             'gender': user.gender,
                             'age': age,
                             'height': user.height,
                             'weight': user.weight
                         })

@app.route('/predict_stroke', methods=['POST'])
@login_required
def predict_stroke():
    if request.method == 'POST':
        try:
            user = User.query.get(session['user_id'])
            
            age = user.calculate_age()

            height = float(request.form.get('height', user.height))
            weight = float(request.form.get('weight', user.weight))
            gender = request.form.get('gender', user.gender)
            
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            residence_type = request.form['residence']
            glucose = float(request.form['glucose'])
            smoking_status = request.form['smoking_status']

            height_m = height / 100
            bmi = weight / (height_m ** 2)
            bmi = round(bmi, 2)

            age = int(request.form.get('age', age))

            input_data = pd.DataFrame({
                'gender': [1 if gender == 'female' else 0],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'avg_glucose_level': [round(glucose, 2)],
                'bmi': [bmi],
                'Residence_type_Rural': [1 if residence_type == 'rural' else 0],
                'Residence_type_Urban': [1 if residence_type == 'urban' else 0],
                'smoking_status_Unknown': [1 if smoking_status == 'unknown' else 0],
                'smoking_status_formerly smoked': [1 if smoking_status == 'formerly_smoked' else 0],
                'smoking_status_never smoked': [1 if smoking_status == 'never_smoked' else 0],
                'smoking_status_smokes': [1 if smoking_status == 'smokes' else 0]
            })

            if model:
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_data)[0][1]
                else:
                    probability = None

                prediction = int(model.predict(input_data)[0])
            else:
                raise Exception("Model not loaded")

            if probability is not None:
                if probability >= 0.4:
                    risk_level = 'High'
                    risk_class = 'high-risk'
                elif probability >= 0.1:
                    risk_level = 'Medium'
                    risk_class = 'medium-risk'
                else:
                    risk_level = 'Low'
                    risk_class = 'low-risk'

                stroke_probability = round(probability * 100, 1)
            else:
                risk_level = 'High' if prediction == 1 else 'Low'
                risk_class = 'high-risk' if prediction == 1 else 'low-risk'
                stroke_probability = None

            patient_data = {
                'name': user.name,
                'email': user.email,
                'age': age,
                'gender': gender,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'residence': residence_type,
                'glucose': glucose,
                'bmi': bmi,
                'height': height,
                'weight': weight,
                'smoking_status': smoking_status
            }

            return render_template('result.html', 
                                prediction=prediction,
                                risk_level=risk_level,
                                risk_class=risk_class,
                                stroke_probability=stroke_probability,
                                patient_data=patient_data,
                                user=user)

        except Exception as e:
            return render_template('stroke.html', 
                                error=f"An error occurred: {str(e)}",
                                user=user)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)