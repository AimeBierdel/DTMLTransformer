import torch
import torch.nn as nn
from numpy import sqrt
import numpy as np
torch.manual_seed(0)

  
class ContextEncoder(nn.Module):
    """_summary_
    Encoder structure used on a list of stock time series in order to generate a context vector. 
    Input is an individual stock features, indexed by t. Dimensions are NstocksxTxN_features
    
    Output is a normalized list of context vector of size Nstocksx Hidden_size
    """
    def __init__(
        self,
        d,
        h,
        n_features, 
        T,
        batch_size,
        transf_size,
        dropout = 0.1,
        n_layers=1
    ):
        super(ContextEncoder, self).__init__()
        
        ### Parameters 
        
        self.d = d
        self.h = h 
        self.n_features = n_features
        self.T = T
        self.batch_size = batch_size
        self.transf_size = transf_size 
        self.n_layers = n_layers
        self.dropout = dropout
        ### Function 
        
        
        
        ### Modules
        self.transf = nn.Linear(self.n_features,self.transf_size )
        self.tanh = nn.Tanh() 
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim =1)
        #self.norm2 = nn.LayerNorm([self.d, self.h])
        #self.norm1 = nn.LayerNorm([self.T, self.h])
        
        self.contextNorm = ContextNorm(2* self.d * self.h * self.n_layers)

        self.lstm = nn.LSTM(
            input_size=self.transf_size,
            hidden_size=self.h,
            bidirectional = True,
            num_layers=self.n_layers,
            batch_first = True, 
        )

    def forward(self, X):
        ### First we will summarize each feature of an individual stock into a transf_size dimensional vector (default 1)
        out = self.transf(X)
        out = self.tanh(out)
        ### out is Batch_size x T x Nstocks x transf_size
        
        ### Then we will apply a LSTM to each stock over the period, feeding LSTM with the 
        ### state from last period. 
        b_size = out.shape[0]
        out = torch.swapaxes(out,1,2)
        
        ### out is Batch_size x Nstocks xT x  transf_size
        
        
        out = torch.flatten(out, start_dim = 0, end_dim = 1)
        
        ### out is now (batchsizexnstocks) x T x transf_size
        for t in range(self.T): 
            stepInput = out[:,t,:]
            stepInput = torch.unsqueeze(stepInput,dim = 1)    
            if t == 0: 
                pred, (h, c) = self.lstm(stepInput)
                htemp = torch.swapaxes(h,0,1)
                hvec = torch.flatten(htemp, start_dim = 1,end_dim=2).unsqueeze(1)
            else : 
                pred,(h,c) = self.lstm(stepInput,(h,c))
                htemp = torch.swapaxes(h,0,1)
                htemp = torch.flatten(htemp, start_dim = 1,end_dim=2).unsqueeze(1)
                hvec = torch.cat( (hvec, htemp) , dim = 1)
        
        ## hvec is a tensor with dimensions (batch_size x nstocks )x T x (2 * hidden_size* n_layers )
        ## Now we will create the attention measure on that context vector for each stock 
        hT = hvec[:, self.T-1,:] #(batch_size x nstocks )x  (2 * hidden_size * n_layers)
        hT = torch.unsqueeze(hT , 2)   #(batch_size x nstocks )x 1 x  (2 * hidden_size* n_layers )
        wgts = torch.matmul(hvec , hT)
        wgts = self.soft(wgts)# (batch_size x T x 1) weights for each h_t 
        out = torch.matmul(hvec.transpose(1,2),wgts)
        ## We convert it back to batch_size x nstocks x (2 * hidden_size *n_layers)
        out = out.view((b_size, self.d,2*self.h*self.n_layers ))
        ### We obtain out, a tensor of size batch_size x nstocks x (2 * hidden_size*n_layers), a vector of contexts for each stock.
        ### We now need to normalize the context vectors across stocks
        out = torch.flatten(out,start_dim=1, end_dim = 2)
        out = self.contextNorm(out)
        out = out.view((b_size, self.d,2*self.h*self.n_layers ))
        return out
        
        
        
class contextAggregator(nn.Module ):
    """
    This class takes local and global context vectors and aggregates them into the multi-level context matrix
    
    Inputs : Local context matrix, global context matrix, correlation parameter 
    
    Output : Context matrix
    """     
    def __init__(
        self,
        beta
    ): 
        super(contextAggregator, self).__init__()
        self.beta = beta
        
    def aggContext(self,localContext,globalContext ):
        return  localContext + self.beta * globalContext.expand(localContext.size() )
    
    
class DASATransformer(nn.Module):
    """_summary_
    Uses a matrix of context with a transformer encoder to produce multivariate forecasts
    Args:
        nn (_tensor_): 
    """
    def __init__(
        self,
        d, 
        h,
        dropout = 0.2
    ):
        super(DASATransformer, self).__init__()
       
       ### Functions / Parameters
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()
        self.d = d
        self.h = h
        self.dropout = dropout 
        
        ### Models 
        self.n1 =  nn.LayerNorm([self.d, self.h])
        self.n2 = nn.LayerNorm([self.d, self.h])
        self.att = nn.MultiheadAttention( embed_dim=self.h, num_heads= 2, dropout = self.dropout )
        self.mlp =  nn.Sequential(
            nn.Linear(self.h, self.h*4),
            nn.LeakyReLU(),
            nn.Linear(self.h*4 , self.h*4),
            nn.LeakyReLU(),
            nn.Linear(self.h*4, self.h),
            nn.LeakyReLU()
        )
        
    def forward(self, X):

        ### Apply attention model 
        out, w = self.att(X, X, X)
        ### Normalization
        out = self.n1(out)
        ### residual connection
        out = out + X + self.mlp( out+X)
        ### Layer Normalization
        out = self.ReLU(out)
        out = self.n2(out)
        
        return out

        
      
        
        
                
class DTMLModel(nn.Module):
    """
    The model that combines individual context computation, aggregation and data-axis transformation
    
    """ 
    def __init__(
        self,
        d,
        T,
        h,
        n_features,
        beta, 
        batch_size,
        nglobal = 1,
        transf_size= -1,
        n_layers=1,
        dropout = 0.2
    ):   
        super(DTMLModel, self).__init__()
        ## Data 
        
        
        ## parameters 
       
        self.d = d - nglobal
        self.T = T
        self.h = h
        self.n_features = n_features
        self.beta = beta
        self.batch_size = batch_size
        self.dglobal = nglobal
        if transf_size < 1 : 
            self.transf_size = self.n_features
        self.n_layers = n_layers
        self.dropout = dropout

   
        ## Functions 
        
        self.sig = nn.Sigmoid()
         

        ### Modules 
       
        # local context encoder
        self.localContextEncoder = ContextEncoder(d = self.d,
        h = self.h,
        n_features = self.n_features, 
        T = self.T,
        batch_size = self.batch_size, 
        transf_size = self.transf_size,
        n_layers = self.n_layers
        )
        #global context encoder
        self.globalContextEncoder = ContextEncoder(self.dglobal,
        self.h,
        self.n_features, 
        self.T, 
        self.batch_size,
        transf_size = self.transf_size,
        n_layers = self.n_layers
        )
        # Context aggregator
        self.contextAg = contextAggregator(self.beta)
        # Dasatransformer
        self.dasaTransformer = DASATransformer(self.d, 2*self.h*self.n_layers, self.dropout)
        
        self.lin = nn.Linear(2*self.h * self.n_layers, 1)
    
    def forward(self,X):
        globX = X[:,:,0:self.dglobal,:]
        locX = X[:,:,self.dglobal : X.shape[2],:]
        lout = self.localContextEncoder(locX)
        gout = self.globalContextEncoder(globX) 
        out = self.contextAg.aggContext(lout,gout)
        out = self.dasaTransformer(out)
        out = self.lin(out)
        #out = self.sig(out)
        return out
    
class ContextNorm(nn.Module):
    """_summary_
        Context normalization as described in the paper.
    Args:
        nn (_type_): A batch_size x n dimension tensor
    """
    def __init__(
        self,
        dim
    ):  
        super(ContextNorm, self).__init__()
        self.dim= dim
        self.diag = DiagonalLayer(dim)
    
    def forward(self,X):
        m = torch.mean(X, 1).unsqueeze(1)
        s = torch.std(X,1).unsqueeze(1)
   
        out = (X - m) / s 
        out = self.diag(out)
        return out
            
class DiagonalLayer(nn.Module):
    """ Custom Linear layer with only diagonal terms """
    def __init__(self, size):
        super().__init__()
        self.size = size
        weights = torch.Tensor(size)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.uniform_(self.weights, -sqrt(5), sqrt(5)) # weight init
        bound = 1 / sqrt(5)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        D = torch.diag(self.weights.t(), diagonal = 0)
        w_times_x= torch.mm(x, D)
        return torch.add(w_times_x, self.bias)  # w times x + b