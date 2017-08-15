# -*- coding: utf-8 -*-
import numpy as np
from vasc import vasc
from helpers import clustering,measure,print_2D
from config import config

if __name__ == '__main__':
    DATASET = 'biase' #sys.argv[1]
    PREFIX = 'biase' #sys.argv[2]
    
    filename = DATASET+'.txt'
    data = open( filename )
    head = data.readline().rstrip().split()
    
    label_file = open( DATASET+'_label.txt' )
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        label_dict[temp[0]] = temp[1]
    label_file.close()
    
    label = []
    for c in head:
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)
    
    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value:idx for idx,value in enumerate(label_set)}
    id_map = {idx:value for idx,value in enumerate(label_set)}
    label = np.asarray( [ name_map[name] for name in label ] )
    
    expr = []
    for line in data:
        temp = line.rstrip().split()[1:]
        temp = [ float(x) for x in temp]
        expr.append( temp )
    
    expr = np.asarray(expr).T
    n_cell,_ = expr.shape
    if n_cell > 150:
        batch_size=config['batch_size']
    else:
        batch_size=32 
    #expr = np.exp(expr) - 1 
    #expr = expr / np.max(expr)

#    
#    percentage = [0.5]
#    
#    for j in range(1):
#        print(j)
#        p = percentage[j]
#        samples = np.random.choice( n_cell,size=int(n_cell*p),replace=True )
#        expr_train = expr[ samples,: ]
#        label_train = label[samples]
    
    #latent = 2
    for i in range(1):
        print("Iteration:"+str(i))
        res = vasc( expr,var=False,
                    latent=config['latent'],
                    annealing=False,
                    batch_size=batch_size,
                    prefix=PREFIX,
                    label=label,
                    scale=config['scale'],
                    patience=config['patience'] 
                )
#            res_file = PREFIX+'_res.h5'
#            res_data = h5py.File( name=res_file,mode='r' )
#            dim2 = res_data['RES5']
#            print(np.max(dim2))
        
        print(res.shape)
        k = len( np.unique(label) )
        cl,_ = clustering( res,k=k)
        dm = measure( cl,label )
        
#            res_data.close()
        ### analysis results
        # plot loss
        
        # plot 2-D visulation
        fig = print_2D( points=res,label=label,id_map=id_map )
#        fig.savefig('embryo.eps')
#        fig = print_2D( points=res_data['RES5'],label=label,id_map=id_map )
#        fig.show()
#        res_data.close()
#        time.sleep(30)
        #res_data.close()
    # plot NMI,ARI curve
#    
#    pollen = h5py.File( name=DATASET+'_'+str(latent)+'_.h5',mode='w' )
#    pollen.create_dataset( name='NMI',data=nmi)
#    pollen.create_dataset( name='ARI',data=ari )
#    pollen.create_dataset( name='HOM',data=hom )
#    pollen.create_dataset( name='COM',data=com )
#    pollen.close()
#    
    
    #print("============SUMMARY==============")
    #k = len(np.unique(label))
    #for r in res:
    #    print("======"+str(r.shape[1])+"========")
    #    pred,si = clustering( r,k=k )
    #    if label is not None:
    #        metrics = measure( pred,label )
    #