import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as SS

def fld(X, Y, n=2):
    '''Implementation of FLD for n components
    (default usage being a 2D data for our plot)'''

    # convert to np arrays
    #X = X.values; Y = Y.values
    #  Y = y
    
    classes = np.unique(Y)
    num_classes = len(classes)
    num_features = X.shape[1]
    
    # Calculate class means:
    class_means = np.array([np.mean(X[Y == c], axis=0) for c in classes])
    
    # Calculate overall mean:
    overall_mean = np.mean(X, axis=0)
    
    # Calculate between-class scatter matrix:
    S_B = np.sum([len(X[Y == c]) * np.outer((mean - overall_mean), 
                                            (mean - overall_mean)) 
                for mean, c in zip(class_means, classes)], axis=0)

    # Compute the within class scatter matrix:
    S_W = np.zeros((num_features, num_features))
    for i, c in enumerate(classes):
        mean_c = class_means[i]
        X_c = X[Y == c]  # Samples belonging to class c
        # Sum of outer products (x - mean_c)(x - mean_c).T over all x in class c
        S_W += np.sum([np.outer((x - mean_c), (x - mean_c)) for x in X_c], axis=0)
    
    # Solve generalized eigenvalue problem S_W^{-1} * S_B * W = lambda * W
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    
    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select first n eigenvs & project data onto subspace spanned by these
    W = eigenvectors[:, :n]; X_lda = X.dot(W)
    
    return X_lda

def pca(X, n=2):
    '''Implementation of PCA for n components 
    (default usage being a 2D data for our plot)'''

    # Center data around mean:
    mean = np.mean(X, axis=0)
    cData = X - mean
    
    # Use np to find covariance, eig values & vectors:
    cov_matrix = np.cov(cData, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigs & select best n components:
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :n]

    # project our data onto said best eig vectors:
    reduced_data = np.dot(cData, top_eigenvectors)
    
    return reduced_data

#################
# READ-IN DATA: #
#################
# read in data & drop labels:
data = pd.read_csv("pima.tr", delim_whitespace=True)
X_train = data.drop(columns=["type"])

# Convert y-vals into a binary: [Yes=1, No=0]
y_train = (data["type"] == "Yes").astype(int)

###################
# SCALE THE DATA: #
###################
scaler = SS()
X_scaled = scaler.fit_transform(X_train)
#y_scaled = scaler.fit_transform(y_train)

########################
# PERFORM FLD ON DATA: #
########################
# Perform PCA & create scatter plot of top two principal components
X_fld = fld(X_scaled, y_train)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_fld[:, 0], X_fld[:, 1], c=y_train, 
                      cmap=plt.cm.get_cmap("coolwarm", 2), edgecolor='k')
cb = plt.colorbar(scatter,label="No                    "\
                  "                            Yes")
cb.set_ticks([0, 1]); cb.set_ticklabels(["No", "Yes"])
plt.clim(-0.5, 1.5)
plt.title('Principle Component Analysis (PCA) of Pima.tr')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('FLD.png'); plt.figure()

########################
# PERFORM PCA ON DATA: #
########################
# Perform PCA & create scatter plot of top two principal components
X_pca = pca(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, 
                      cmap=plt.cm.get_cmap("coolwarm", 2), edgecolor='k')
cb = plt.colorbar(scatter,label="No                    "\
                  "                            Yes")
cb.set_ticks([0, 1]); cb.set_ticklabels(["No", "Yes"])
plt.clim(-0.5, 1.5)
plt.title('Principle Component Analysis (PCA) of Pima.tr')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('PCA.png'); plt.figure()

##########################
# PERFORM t-SNE ON DATA: #
##########################
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Perform t-SNE & now create a scatterplot of it's determined two components:
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, 
                      cmap=plt.cm.get_cmap("coolwarm", 2), edgecolor='k')
cb = plt.colorbar(scatter,label="No                  "\
                  "                    Yes")
cb.set_ticks([0, 1]); cb.set_ticklabels(["No", "Yes"])
plt.clim(-0.5, 1.5)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization")
plt.savefig('tSNE.png')