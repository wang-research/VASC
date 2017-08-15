import numpy as np
import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt

#plt.ioff()

import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,homogeneity_score,completeness_score,silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope


markers = {',': 'pixel', 'o': 'circle','*': 'star', 'v': 'triangle_down',
           '^': 'triangle_up', '<': 'triangle_left', '>': 'triangle_right', 
           '1': 'tri_down', '2': 'tri_up', '3': 'tri_left', '4': 'tri_right', 
           '8': 'octagon', 's': 'square', 'p': 'pentagon', 
           'h': 'hexagon1', 'H': 'hexagon2', '+': 'plus', 'x': 'x', '.': 'point', 
           'D': 'diamond', 'd': 'thin_diamond', '|': 'vline', '_': 'hline',
           'P': 'plus_filled', 'X': 'x_filled', 0: 'tickleft', 
           1: 'tickright', 2: 'tickup', 3: 'tickdown', 4: 'caretleft', 5: 'caretright',
           6: 'caretup', 7: 'caretdown', 8: 'caretleftbase', 9: 'caretrightbase', 10: 'caretupbase',
           11: 'caretdownbase', 'None': 'nothing', None: 'nothing', ' ': 'nothing', '': 'nothing'}
markers_keys = list(markers.keys())[:20]

font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 30}

mpl.rc('font', **font)

sns.set_style("ticks")

colors = ["windows blue", "amber", 
          "greyish", "faded green", 
          "dusty purple","royal blue","lilac",
          "salmon","bright turquoise",
          "dark maroon","light tan",
          "orange","orchid",
          "sandy","topaz",
          "fuchsia","yellow",
          "crimson","cream"
          ]
current_palette = sns.xkcd_palette(colors)

def print_2D( points,label,id_map ):
    '''
    points: N_samples * 2
    label: (int) N_samples
    id_map: map label id to its name
    '''  
    fig = plt.figure()
    #current_palette = sns.color_palette("RdBu_r", max(label)+1)
    n_cell,_ = points.shape
    if n_cell > 500:
        s = 10
    else:
        s = 20
    
    ax = plt.subplot(111)
    print( np.unique(label) )
    for i in np.unique(label):
        ax.scatter( points[label==i,0], points[label==i,1], c=current_palette[i], label=id_map[i], s=s,marker=markers_keys[i] )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
        
    ax.legend(scatterpoints=1,loc='upper center',
              bbox_to_anchor=(0.5,-0.08),ncol=6,
              fancybox=True,
              prop={'size':8}
              )
    sns.despine()
    return fig

def print_heatmap( points,label,id_map ):
    '''
    points: N_samples * N_features
    label: (int) N_samples
    id_map: map label id to its name
    '''
    # = sns.color_palette("RdBu_r", max(label)+1)
    #cNorm = colors.Normalize(vmin=0,vmax=max(label)) #normalise the colormap
    #scalarMap = cm.ScalarMappable(norm=cNorm,cmap='Paired') #map numbers to colors
    
    index = [id_map[i] for i in label]
    df = DataFrame( 
            points,
            columns = list(range(points.shape[1])),
            index = index
            )
    row_color = [current_palette[i] for i in label]
    
    cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
    g = sns.clustermap( df,cmap=cmap,row_colors=row_color,col_cluster=False,xticklabels=False,yticklabels=False) #,standard_scale=1 )
    
    return g.fig

def measure( predicted,true ):
    NMI = normalized_mutual_info_score( true,predicted )
    print("NMI:"+str(NMI))
    RAND = adjusted_rand_score( true,predicted )
    print("RAND:"+str(RAND))
    HOMO = homogeneity_score( true,predicted )
    print("HOMOGENEITY:"+str(HOMO))
    COMPLETENESS = completeness_score( true,predicted )
    print("COMPLETENESS:"+str(COMPLETENESS))
    return {'NMI':NMI,'RAND':RAND,'HOMOGENEITY':HOMO,'COMPLETENESS':COMPLETENESS}

def clustering( points, k=2,name='kmeans'):
    '''
    points: N_samples * N_features
    k: number of clusters
    '''
    if name == 'kmeans':
        kmeans = KMeans( n_clusters=k,n_init=100 ).fit(points)
        ## print within_variance
        #cluster_distance = kmeans.transform( points )
        #within_variance = sum( np.min(cluster_distance,axis=1) ) / float( points.shape[0] )
        #print("AvgWithinSS:"+str(within_variance))
        if len( np.unique(kmeans.labels_) ) > 1: 
            si = silhouette_score( points,kmeans.labels_ )
            #print("Silhouette:"+str(si))
        else:
            si = 0
            print("Silhouette:"+str(si))
        return kmeans.labels_,si
    
    if name == 'spec':
        spec= SpectralClustering( n_clusters=k,affinity='cosine' ).fit( points )
        si = silhouette_score( points,spec.labels_ )
        print("Silhouette:"+str(si))
        return spec.labels_,si
        
def cart2polar( points ):
    '''
    points: N_samples * 2
    '''
    return np.c_[np.abs(points), np.angle(points)]
        
def outliers_detection(expr):
    x = PCA(n_components=2).fit_transform(expr)
    ee = EllipticEnvelope()
    ee.fit(x)
    oo = ee.predict(x)
    
    return oo
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
        

    