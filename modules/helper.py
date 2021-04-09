import numpy as np

#flatten an 3D np array
def flatten(X, pos = -1):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    if pos == -1:
        pos = X.shape[1]-1
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, pos, :]
    return(flattened_X)

# Scale 3D array. X = 3D array, scalar = scale object from sklearn. Output = scaled 3D array.
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

# Symbolize a 3D array. X = 3D array, scalar = SAX symbolizer object. Output = symbolic 3D string array.
def symbolize(X, scaler):
    X_s = scaler.transform(X)
    #X_s = X.astype('U13') 
    #for i in range(X.shape[0]):
        #X_s[i, :, :][:,0] = scaler.transform(np.array([X[i, :, :][:,0]]))
        #X_t.append(' '.join(X_s[i, :, :][:,0]))
    return X_s

# maps a string from vocab to [-1,1] in uniform steps
def trans(val, vocab) -> float:
    for i in range(len(vocab)):
        if val == vocab[i]:
            halfSize = (len(vocab)-1)/2
            return (i - halfSize) / halfSize

# Transform the data based on the given already fit SAX scaler and maps the symbols to [-1,1] via the trans method 
def symbolizeTrans(X, scaler, bins = 5):
    vocab = scaler._check_params(bins)
    X_s = scaler.transform(X)
    #X_s = X.astype(str) 
    for i in range(X.shape[0]):
        X = X.astype(float)
        
        #z1 = scaler.transform(np.array([X[i, :, :][:,0]]))
        
        for j in range(X.shape[1]):
            X[i][j] = trans(X_s[i][j], vocab)
    return X

#saves a object
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        np.save(f, obj)

#loades a saved object
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return np.load(f, allow_pickle=True)
    
#truncate a value to 3 zeros
def truncate(n):
    return int(n * 1000) / 1000