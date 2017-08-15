# -*- coding: utf-8 -*-
from keras.layers import Input,Dense,Activation,Lambda,RepeatVector,merge,Reshape,Layer,Dropout,BatchNormalization,Permute
import keras.backend as K
from keras.models import Model
from helpers import measure,clustering,print_2D,print_heatmap,cart2polar,outliers_detection
#from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.utils.layer_utils import print_summary
import numpy as np
from keras.optimizers import RMSprop,Adagrad,Adam
from keras import metrics
from config import config
import h5py

tau = 1.0

def sampling(args):
    epsilon_std = 1.0
    
    if len(args) == 2:
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), 
                              mean=0.,
                              stddev=epsilon_std)
    #
        return z_mean + K.exp( z_log_var / 2 ) * epsilon
    else:
        z_mean = args[0]
        epsilon = K.random_normal(shape=K.shape(z_mean), 
                              mean=0.,
                              stddev=epsilon_std)
        return z_mean + K.exp( 1.0 / 2 ) * epsilon
        
        
def sampling_gumbel(shape,eps=1e-8):
    u = K.random_uniform( shape )
    return -K.log( -K.log(u+eps)+eps )

def compute_softmax(logits,temp):
    z = logits + sampling_gumbel( K.shape(logits) )
    return K.softmax( z / temp )

def gumbel_softmax(args):
    logits,temp = args
    return compute_softmax(logits,temp)

class NoiseLayer(Layer):
    def __init__(self, ratio, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.ratio = ratio

    def call(self, inputs, training=None):
        def noised():
            return inputs * K.random_binomial(shape=K.shape(inputs),
                                              p=self.ratio
                                              )
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(NoiseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        return dict(list(base_config.items()) + list(config.items()))
 
class VASC:
    def __init__(self,in_dim,latent=2,var=False):
        self.in_dim =in_dim
        self.vae = None
        self.ae = None
        self.aux = None
        self.latent = latent
        self.var = var
        
    
    def vaeBuild( self ):
        var_ = self.var
        in_dim = self.in_dim
        expr_in = Input( shape=(self.in_dim,) )
        
        ##### The first part of model to recover the expr. 
        h0 = Dropout(0.5)(expr_in) 
        ## Encoder layers
        h1 = Dense( units=512,name='encoder_1',kernel_regularizer=regularizers.l1(0.01) )(h0)
        h2 = Dense( units=128,name='encoder_2' )(h1)
        h2_relu = Activation('relu')(h2)
        h3 = Dense( units=32,name='encoder_3' )(h2_relu)
        h3_relu = Activation('relu')(h3)

        
        z_mean = Dense( units= self.latent ,name='z_mean' )(h3_relu)
        if self.var:
            z_log_var = Dense( units=2,name='z_log_var' )(h3_relu)
            z_log_var = Activation( 'softplus' )(z_log_var)
       
                    
        ## sampling new samples
            z = Lambda(sampling, output_shape=(self.latent,))([z_mean,z_log_var])
        else:
            z = Lambda(sampling, output_shape=(self.latent,))([z_mean])
        
        ## Decoder layers
        decoder_h1 = Dense( units=32,name='decoder_1' )(z)
        decoder_h1_relu = Activation('relu')(decoder_h1)
        decoder_h2 = Dense( units=128,name='decoder_2' )(decoder_h1_relu)
        decoder_h2_relu = Activation('relu')(decoder_h2)  
        decoder_h3 = Dense( units=512,name='decoder_3' )(decoder_h2_relu)
        decoder_h3_relu = Activation('relu')(decoder_h3)
        expr_x = Dense(units=self.in_dim,activation='sigmoid')(decoder_h3_relu)

        
        expr_x_drop = Lambda( lambda x: -x ** 2 )(expr_x)
        #expr_x_drop_log = merge( [drop_ratio,expr_x_drop],mode='mul' )  ###  log p_drop =  log(exp(-\lambda x^2))
        expr_x_drop_p = Lambda( lambda x:K.exp(x) )(expr_x_drop)
        expr_x_nondrop_p = Lambda( lambda x:1-x )( expr_x_drop_p )
        expr_x_nondrop_log = Lambda( lambda x:K.log(x+1e-20) )(expr_x_nondrop_p)
        expr_x_drop_log = Lambda( lambda x:K.log(x+1e-20) )(expr_x_drop_p)        
        expr_x_drop_log = Reshape( target_shape=(self.in_dim,1) )(expr_x_drop_log)
        expr_x_nondrop_log = Reshape( target_shape=(self.in_dim,1) )(expr_x_nondrop_log)     
        logits = merge( [expr_x_drop_log,expr_x_nondrop_log],mode='concat',concat_axis=-1 )
        
        temp_in = Input( shape=(self.in_dim,) )
        temp_ = RepeatVector( 2 )(temp_in)
        print(temp_.shape)
        temp_ = Permute( (2,1) )(temp_)
        samples = Lambda( gumbel_softmax,output_shape=(self.in_dim,2,) )( [logits,temp_] )          
        samples = Lambda( lambda x:x[:,:,1] )(samples)
        samples = Reshape( target_shape=(self.in_dim,) )(samples)      
##        #print(samples.shape)
        
        out = merge( [expr_x,samples],mode='mul' )

        class VariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLayer, self).__init__(**kwargs)
        
            def vae_loss(self, x, x_decoded_mean):
                xent_loss = in_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                if var_:
                    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                else:
                    kl_loss = - 0.5 * K.sum(1 + 1 - K.square(z_mean) - K.exp(1.0), axis=-1)
                return K.mean(xent_loss + kl_loss)
        
            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x
        
        y = VariationalLayer()([expr_in, out])
        vae = Model( inputs= [expr_in,temp_in],outputs=y )
        
        opt = RMSprop( lr=0.001 )
        vae.compile( optimizer=opt,loss=None )
        
        ae = Model( inputs=[expr_in,temp_in],outputs=[ h1,h2,h3,h2_relu,h3_relu,
                                                       z_mean,z,decoder_h1,decoder_h1_relu,
                                                       decoder_h2,decoder_h2_relu,decoder_h3,decoder_h3_relu,
                                                       samples,out
                                                       ] )
        aux = Model( inputs=[expr_in,temp_in],outputs=[out] )
        
        self.vae = vae
        self.ae = ae
        self.aux = aux

def vasc( expr,
          epoch = 5000,
          latent=2,
          patience=50,
          min_stop=500,
          batch_size=32,
          var = False,
          prefix='test',
          label=None,
          log=True,
          scale=True,
          annealing=False,
          tau0 = 1.0,
          min_tau = 0.5,
          rep=0):
    '''
    VASC: variational autoencoder for scRNA-seq datasets
    
    ============
    Parameters:
        expr: expression matrix (n_cells * n_features)
        epoch: maximum number of epochs, default 5000
        latent: dimension of latent variables, default 2
        patience: stop if loss showes insignificant decrease within *patience* epochs, default 50
        min_stop: minimum number of epochs, default 500
        batch_size: batch size for stochastic optimization, default 32
        var: whether to estimate the variance parameters, default False
        prefix: prefix to store the results, default 'test'
        label: numpy array of true labels, default None
        log: if log-transformation should be performed, default True
        scale: if scaling (making values within [0,1]) should be performed, default True
        annealing: if annealing should be performed for Gumbel approximation, default False
        tau0: initial temperature for annealing or temperature without annealing, default 1.0
        min_tau: minimal tau during annealing, default 0.5
        rep: not used
    
    =============
    Values:
        point: dimension-*latent* results
        A file named (*prefix*_*latent*_res.h5): we prefer to use this file to analyse results to the only return values.
        This file included the following keys:
            POINTS: all intermediated latent results during the iterations
            LOSS: loss values during the training procedure
            RES*i*: i from 0 to 14
                - hidden values just for reference
        We recommend use POINTS and LOSS to select the final results in terms of users' preference.
    '''
    
    
    expr[expr<0] = 0.0

    if log:
        expr = np.log2( expr + 1 )
    if scale:
        for i in range(expr.shape[0]):
            expr[i,:] = expr[i,:] / np.max(expr[i,:])
   
#    if outliers:
#        o = outliers_detection(expr)
#        expr = expr[o==1,:]
#        if label is not None:
#            label = label[o==1]
    
    
    if rep > 0:
        expr_train = np.matlib.repmat( expr,rep,1 )
    else:
        expr_train = np.copy( expr )
    
    vae_ = VASC( in_dim=expr.shape[1],latent=latent,var=var )
    vae_.vaeBuild()
    #print_summary( vae_.vae )
    
    points = []
    loss = []
    prev_loss = np.inf
    #tau0 = 1.
    tau = tau0
    #min_tau = 0.5
    anneal_rate = 0.0003
    for e in range(epoch):
        cur_loss = prev_loss
        
        #mask = np.ones( expr_train.shape,dtype='float32' )
        #mask[ expr_train==0 ] = 0.0
        if e % 100 == 0 and annealing:
            tau = max( tau0*np.exp( -anneal_rate * e),min_tau   )
            print(tau)

        tau_in = np.ones( expr_train.shape,dtype='float32' ) * tau
        #print(tau_in.shape)
        
        loss_ = vae_.vae.fit( [expr_train,tau_in],expr_train,epochs=1,batch_size=batch_size,
                             shuffle=True,verbose=0
                             )
        train_loss = loss_.history['loss'][0]
        cur_loss = min(train_loss,cur_loss)
        loss.append( train_loss )
        #val_loss = -loss.history['val_loss'][0]
        res = vae_.ae.predict([expr,tau_in])
        points.append( res[5] )
        if label is not None:
            k=len(np.unique(label))
            
        if e % patience == 1:
            print( "Epoch %d/%d"%(e+1,epoch) )
            print( "Loss:"+str(train_loss) )
            if abs(cur_loss-prev_loss) < 1 and e > min_stop:
                break
            prev_loss = train_loss
            if label is not None:
                try:
                    cl,_ = clustering( res[5],k=k )
                    measure( cl,label )
                except:
                    print('Clustering error')    
                    
    #
    ### analysis results
    #cluster_res = np.asarray( cluster_res )
    points = np.asarray( points )
    aux_res = h5py.File( prefix+'_'+str(latent)+'_res.h5',mode='w' )
    #aux_res.create_dataset( name='EXPR',data=expr )
    #aux_res.create_dataset( name='CLUSTER',data=cluster_res )
    aux_res.create_dataset( name='POINTS',data=points )
    aux_res.create_dataset( name='LOSS',data=loss )
    count = 0
    for r in res:
        aux_res.create_dataset( name='RES'+str(count),data=r)
        count += 1
    aux_res.close()
    
    return res[5]
    
    
    
    