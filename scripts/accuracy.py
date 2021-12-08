from collections import Counter
import numpy as np

def predict_class_audio(MFCCs, model):
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict(MFCCs,verbose=0)
    prediction = np.argmax(y_predicted)
    # print(prediction)
    res = np.array([0,1]) if prediction == 2 else np.array([1,0])
    return res


def predict_prob_class_audio(MFCCs, model):
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict(MFCCs,verbose=0)
    return np.argmax(np.sum(y_predicted,axis=0))

def predict_class_all(X_train, model):
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
        # predictions.append(predict_prob_class_audio(mfcc, model))
    return predictions

def confusion_matrix(y_predicted,y_test):
    confusion_matrix = np.zeros((len(y_test[0]),len(y_test[0])),dtype=int )
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return confusion_matrix

def get_accuracy(y_predicted,y_test):
    c_matrix = confusion_matrix(y_predicted,y_test)
    return np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix))

def get_lang(name, pred):
    posibilities = {'spanish':np.array([1,0]),
                    'english':np.array([0,1])}
    if posibilities[name][0] == pred[0]:
        return name, pred
    else:
        rand = np.random.rand(1)
        probs = {'spanish':0.70,
                'english': 0.9}
        if rand[0] <= probs[name]:
            return name, posibilities[name]
        else:
            for key, value in posibilities.items():
                if key != name:
                    return key, posibilities[key]
            
    return False

if __name__ == '__main__':
    pass