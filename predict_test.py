import pandas as pd
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

test_df = pd.read_csv("test.csv")

ids = test_df['id']


test_df = test_df.drop('id', axis=1)


test_df = test_df[features]


test_scaled = scaler.transform(test_df)


predictions = model.predict(test_scaled)

output = pd.DataFrame({
    "id": ids,
    "price_range": predictions
})

output.to_csv("submission.csv", index=False)

print(" Predictions saved to submission.csv")