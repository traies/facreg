from sklearn import svm

def print_predict(trainproj, testproj, class_list, testl):
    print("----------------------------------------")
    print("Comparacion sujeto esperado con obtenido")
    print("----------------------------------------")

    clf = svm.LinearSVC( random_state=0)
    clf.fit(trainproj, class_list)

    predict_list = clf.predict(testproj)

    for i in range(0, predict_list.size):
        print("Sujeto esperado s{0} | obtenido: s{1}".format(testl[i], predict_list[i]))

    score = clf.score(testproj, testl)
    print("Porcentaje de acierto {0}%".format(score*100))


def predict_all(trainproj, testproj, class_list, testl):
    clf = svm.LinearSVC(random_state=0)
    clf.fit(trainproj, class_list)

    return clf.score(testproj, testl)

def print_predict_all(trainproj, testproj, class_list, testl):
    print("---------------------------------------------------------------")
    print("Relacion entre cantidad de autocaras y porcentaje de acierto")
    print("---------------------------------------------------------------")

    g = [[],[]]
    for i in range(1, trainproj.shape[1] + 1):
        aux = predict_all(trainproj[:, 0:i], testproj[:, 0:i], class_list, testl)
        g[0].append(i)
        g[1].append(aux*100)
        print("usando {0} autocaras: {1}%".format(i, aux*100))

    return g




