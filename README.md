# hackathon-brain-stroke


A **machine learning-powered** web application to predict the risk of **brain stroke** based on user health inputs.  

---

## 📌 Features  

✅ User Authentication (Signup/Login)  
✅ Stroke Risk Prediction using **RandomForestClassifier**  
✅ Interactive Form for Input Data  
✅ Dynamic Result Display  
✅ Profile Management  

---

## 🚀 Tech Stack  

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask  
- **Machine Learning Model:** RandomForestClassifier (saved as `stroke_model.joblib`)  
- **Database:** SQLite3  
- **Dataset:** `brain_stroke.csv`  

---

## 📂 Project Structure  

```
/hackathon-brain-stroke
│── static/                # Static files (CSS, JS)
│── templates/             # HTML templates
│── app.py                 # Main Flask application
│── save_model.py          # Model training script
│── stroke_model.joblib    # Pre-trained ML model
│── brain_stroke.csv       # Dataset used for training
│── requirements.txt       # Dependencies
│── edit_profile.html      # Edit user profile page
│── login.html             # Login page
│── profile.html           # User profile
│── result.html            # Prediction result page
│── signup.html            # Signup page
│── stroke.html            # Stroke prediction form
```

---

## 🔮 Future Enhancements  

🔹 Improve model accuracy with deep learning  
🔹 Deploy the project online (e.g., Render, Vercel, or AWS)  
🔹 Add real-time health monitoring features  

---
