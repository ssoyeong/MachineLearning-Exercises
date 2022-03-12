# wk7_PHW_201835518_전소영

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

flu = ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'Y', 'Y', 'N']
fever = ['L', 'M', 'H', 'M', 'L', 'M', 'H', 'L', 'L', 'M', 'M', 'H', 'L', 'M', 'H', 'M','L', 'H', 'M', 'M','L', 'M', 'H', 'M','L', 'M', 'H', 'M','L', 'M']
sinus = ['Y', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N', 'Y', 'Y','Y', 'N']
ache = ['Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N','Y', 'N', 'N', 'N', 'Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N', 'N', 'N','Y', 'N']
swell = ['Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N', 'Y', 'N','Y', 'N']
headache = ['N', 'N', 'Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N', 'Y', 'Y', 'N', 'N','Y', 'Y', 'N', 'N']


def doNB(features, test):
    # Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(features, label)
    # Predict
    predicted = model.predict(test)
    # print('Predicted Value : ', predicted)

    return predicted


# Create labelEncoder
le = preprocessing.LabelEncoder()

# Convert string labels into numbers
label = le.fit_transform(flu)
fever_encoded = le.fit_transform(fever)
sinus_encoded = le.fit_transform(sinus)
ache_encoded = le.fit_transform(ache)
swell_encoded = le.fit_transform(swell)
headache_encoded = le.fit_transform(headache)

features = list(zip(fever_encoded, sinus_encoded, ache_encoded, swell_encoded, headache_encoded))
# print(features)

test = [[0, 1, 1, 1, 0]]  # fever:H, sinus:Y, ache:Y, swell:Y, headache:N
# Predict using GaussianNB
predicted = doNB(features, test)

# Print result
print("Input:", test, "(fever:H, sinus:Y, ache:Y, swell:Y, headache:N)")
if(predicted == 1):
    print('Predicted flu')
else:
    print('Predicted Not flu')
