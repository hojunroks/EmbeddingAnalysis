
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def featureToTSNE(features, labels, savePath):
    tsne = TSNE(n_components=2)
    results = tsne.fit_transform(features)
    resultsDF = pd.DataFrame(data = results, columns = ['tsne dim 1', 'tsne dim 2'])
    colors = ['gold','aquamarine','r','b','g','hotpink','aqua','mediumorchid','c','m','y','k','lightcoral','royalblue','greenyellow']
    cs_colors = [colors[labels[i]] for i in range(len(labels))]
    ids = [0,1,2,3,4,5,6,7,8,9]
    label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(30)
    ts = fig.add_subplot(1, 1, 1)
    ts.set_xlabel('Dimension 1', fontsize = 15)
    ts.set_ylabel('Dimension 2', fontsize = 15)
    ts.set_title('TSNE', fontsize = 20)

    ts.scatter(resultsDF['tsne dim 1'],
        resultsDF['tsne dim 2'], 
        c = cs_colors
    )
    recs=[]
    for i in range(len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    ts.legend(handles=recs, labels=label_names, title="cell")
    ts.grid()
    plt.savefig(savePath)