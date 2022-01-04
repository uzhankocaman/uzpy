def knn_classifier(likelihoods, labels, Ns, data):
    # estimate label of new data with KNN
    # Input
    # densities           : estimated density through knn_density_estimator
    # labels              : name of the classes
    # data                : new datapoints
    # Output
    # predicted_class     : predicted label of the new datapoints
    
    probability = {}
    
    for likelihood, label, N in zip(likelihoods, labels, Ns):
        try: 
            probability[label] = likelihood(data)*N
        except ValueError:
            probability[label] = 10**(-6)
    
    predicted_class = max(probability, key=probability.get) 
    return predicted_class