import pickle

with open('models/tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/rf_baseline.pkl', 'rb') as f:
    rf_baseline = pickle.load(f)

sample_real_text = """WEST PALM BEACH, Fla./WASHINGTON (Reuters) - The White House said on Friday it was set to kick off talks next week with Republican and Democratic congressional leaders on immigration policy, government spending and other issues that need to be wrapped up early in the new year. The expected flurry of legislative activity comes as Republicans and Democrats begin to set the stage for midterm congressional elections in November. President Donald Trump's Republican Party is eager to maintain control of Congress while Democrats look for openings to wrest seats away in the Senate and the House of Representatives. On Wednesday, Trump's budget chief Mick Mulvaney and legislative affairs director Marc Short will meet with Senate Majority Leader Mitch McConnell and House Speaker Paul Ryan - both Republicans - and their Democratic counterparts, Senator Chuck Schumer and Representative Nancy Pelosi, the White House said. That will be followed up with a weekend of strategy sessions for Trump, McConnell and Ryan on Jan. 6 and 7 at the Camp David presidential retreat in Maryland, according to the White House. The Senate returns to work on Jan. 3 and the House on Jan. 8. Congress passed a short-term government funding bill last week before taking its Christmas break, but needs to come to an agreement on defense spending and various domestic programs by Jan. 19, or the government will shut down."""

X = vectorizer.transform([sample_real_text])
print("Classes:", rf_baseline.classes_)
print("Prediction:", rf_baseline.predict(X))
print("Probabilities:", rf_baseline.predict_proba(X))

with open('models/selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

with open('models/rf_pso.pkl', 'rb') as f:
    rf_pso = pickle.load(f)

X_pso = X[:, selected_features]
print("PSO Prediction:", rf_pso.predict(X_pso))
print("PSO Probabilities:", rf_pso.predict_proba(X_pso))