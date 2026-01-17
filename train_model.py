import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 데이터 로드 및 학습
iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# 모델 저장
joblib.dump(model, "iris_model.pkl")
print("모델 저장 완료!")