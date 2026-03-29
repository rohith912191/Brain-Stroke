import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv('brain_stroke.csv')
df['gender'] = [0 if i != 'Female' else 1 for i in df['gender']]
df = pd.get_dummies(df, columns=['Residence_type', 'smoking_status'])

X = df.drop(['stroke'], axis=1)
y = df['stroke']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=3)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

joblib.dump(model, 'stroke_model.joblib')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')
print('Model saved as stroke_model.joblib')