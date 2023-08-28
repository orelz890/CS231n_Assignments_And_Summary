from builtins import range
from matplotlib.cm import scale
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    dim_size = x[0].shape
    X = x.reshape(x.shape[0], np.prod(dim_size))
    out = X.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)
    # input gradient
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    # weight gradient
    dw = X.T.dot(dout)
    # bias gradient
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    layernorm = bn_param.get('layernorm', 0)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta

        cache = (x, x_norm, sample_mean, sample_var, gamma, beta, eps, layernorm)
        
        if not layernorm:
          # Exactly as instructed we are updating the running_mean & running_var
          running_mean = momentum * running_mean + (1 - momentum) * sample_mean
          running_var = momentum * running_var + (1 - momentum) * sample_var

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

        # While testing we dont have to save anything
        cache = None
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################

    # Extract the values needed for backward from cache 
    x, x_norm, sample_mean, sample_var, gamma, beta, eps, layernorm = cache
    N, D = x.shape

    # Compute the gradient of the loss with respect to the shift parameter beta
    dbeta = np.sum(dout, axis=0)
    # Compute the gradient of the loss with respect to the scale parameter gamma
    dgamma = np.sum(dout * x_norm, axis=0)

    # Compute the gradient of the loss with respect to the normalized input x_norm
    dx_norm = dout * gamma

    # Compute the gradient of the loss with respect to the variance of the input x
    dvar = np.sum(dx_norm * (x - sample_mean) * (-0.5) * (sample_var + eps) ** (-1.5), axis=0)

    # Compute the gradient of the loss with respect to the mean of the input x
    dmean = np.sum(dx_norm * (-1 / np.sqrt(sample_var + eps)), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    
    # Compute the gradient of the loss with respect to the inputs x
    dx = dx_norm * (1 / np.sqrt(sample_var + eps)) + dvar * 2 * (x - sample_mean) / N + dmean / N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, x_norm, sample_mean, sample_var, gamma, beta, eps, layernorm = cache
    N = x.shape[0]
    
    # Calculate the gradient of the loss with respect to the shift parameter beta
    dbeta = np.sum(dout, axis=0)
    # Calculate the gradient of the loss with respect to the scale parameter gamma
    dgamma = np.sum(dout * x_norm, axis=0)

    # Calculate the gradient of the loss with respect to the normalized input x_norm. 
    dx_norm = dout * gamma

    # Compute intermediate gradients
    dx_norm_sum = np.sum(dx_norm, axis=0)
    dx_norm_dot_x_norm = np.sum(dx_norm * x_norm, axis=0)

    # The overall expression is scaled by this to account for the normalization and scaling factors.
    overall_scale = (1.0 / N) * (1.0 / np.sqrt(sample_var + eps))

    # scale the gradient with respect to x_norm before applying further computations. 
    # Multiplying dx_norm by N ensures that the gradient is properly scaled to 
    # account for the contribution of each training example in the batch.
    grad_scale = N * dx_norm

    # Compute the gradient of the loss with respect to the input x:
    # Note: The subtraction of dx_norm_sum ensures that the gradient with 
    # respect to x is correctly distributed among the samples.
    # Note2: The dx_norm_dot_x_norm term represents the contribution of x_norm 
    # to the gradient of x. By multiplying dx_norm by dx_norm_dot_x_norm, we 
    # ensure that the gradient is appropriately distributed to the elements of x
    # based on their influence on the output.
    dx = overall_scale * (grad_scale - dx_norm_sum - x_norm * dx_norm_dot_x_norm)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # Step 1: Calculate the mean across the features (D dimension)
    mean = np.mean(x, axis=1, keepdims=True)
    
    # Step 2: Calculate the variance across the features (D dimension)
    var = np.var(x, axis=1, keepdims=True)
    
    # Step 3: Normalize the data
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # Step 4: Scale and shift the normalized data using gamma and beta
    out = gamma * x_norm + beta
    
    # Cache values needed for backward pass
    cache = (x, x_norm, mean, var, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, x_norm, mean, var, gamma, beta, eps = cache
    
    N, D = dout.shape
    
    # Step 1: Calculate dgamma and dbeta:
    # Compute the gradient of the loss with respect to the shift parameter beta
    dbeta = np.sum(dout, axis=0)
    # Compute the gradient of the loss with respect to the scale parameter gamma
    dgamma = np.sum(dout * x_norm, axis=0)
    
    # Step 2: Calculate dx:
    # Compute the gradient of the loss with respect to the normalized input x_norm
    dx_norm = dout * gamma

    # Compute the gradient of the loss with respect to the variance of the input x
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=1, keepdims=True)
    
    # Compute the gradient of the loss with respect to the mean of the input x
    dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=1, keepdims=True) + \
            dvar * np.mean(-2 * (x - mean), axis=1, keepdims=True)
    
    # Compute the gradient of the loss with respect to the inputs x
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mean) / D + dmean / D
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape

    assert (H - FH + 2 * pad) % stride == 0
    assert (W - FW + 2 * pad) % stride == 0
    outH = 1 + (H - FH + 2 * pad) / stride
    outW = 1 + (W - FW + 2 * pad) / stride

    # create output tensor after convolution layer
    out = np.zeros((N, F, outH, outW))

    # padding all input data
    x_pad = np.pad(x, ((0,0), (0,0),(pad,pad),(pad,pad)), 'constant')
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]    

    # create w_row matrix
    w_row = w.reshape(F, C*FH*FW)                            #[F x C*FH*FW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C*FH*FW, outH*outW))                   #[C*FH*FW x H'*W']
    for index in range(N):
        neuron = 0 
        for i in range(0, H_pad-FH+1, stride):
            for j in range(0, W_pad-FW+1,stride):
                x_col[:,neuron] = x_pad[index,:,i:i+FH,j:j+FW].reshape(C*FH*FW)
                neuron += 1
        out[index] = (w_row.dot(x_col) + b.reshape(F,1)).reshape(F, outH, outW)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x_pad, w, b, conv_param = cache
    N, F, outH, outW = dout.shape
    N, C, Hpad, Wpad = x_pad.shape
    FH, FW = w.shape[2], w.shape[3]
    stride = conv_param['stride']
    pad = conv_param['pad']

    # initialize gradients
    dx = np.zeros((N, C, Hpad - 2*pad, Wpad - 2*pad))
    dw, db = np.zeros(w.shape), np.zeros(b.shape)

    # create w_row matrix
    w_row = w.reshape(F, C*FH*FW)                            #[F x C*FH*FW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C*FH*FW, outH*outW))                   #[C*FH*FW x H'*W']
    for index in range(N):
        out_col = dout[index].reshape(F, outH*outW)          #[F x H'*W']
        w_out = w_row.T.dot(out_col)                         #[C*FH*FW x H'*W']
        dx_cur = np.zeros((C, Hpad, Wpad))
        neuron = 0
        for i in range(0, Hpad-FH+1, stride):
            for j in range(0, Wpad-FW+1, stride):
                dx_cur[:,i:i+FH,j:j+FW] += w_out[:,neuron].reshape(C,FH,FW)
                x_col[:,neuron] = x_pad[index,:,i:i+FH,j:j+FW].reshape(C*FH*FW)
                neuron += 1
        dx[index] = dx_cur[:,pad:-pad, pad:-pad]
        dw += out_col.dot(x_col.T).reshape(F,C,FH,FW)
        db += out_col.sum(axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    stride = pool_param['stride']
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    outH = 1 + (H - PH) / stride
    outW = 1 + (W - PW) / stride

    # create output tensor for pooling layer
    out = np.zeros((N, C, outH, outW))
    for index in range(N):
        out_col = np.zeros((C, outH*outW))
        neuron = 0
        for i in range(0, H - PH + 1, stride):
            for j in range(0, W - PW + 1, stride):
                pool_region = x[index,:,i:i+PH,j:j+PW].reshape(C,PH*PW)
                out_col[:,neuron] = pool_region.max(axis=1)
                neuron += 1
        out[index] = out_col.reshape(C, outH, outW)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, outH, outW = dout.shape
    H, W = x.shape[2], x.shape[3]
    stride = pool_param['stride']
    PH, PW = pool_param['pool_height'], pool_param['pool_width']

    # initialize gradient
    dx = np.zeros(x.shape)
    
    for index in range(N):
        dout_row = dout[index].reshape(C, outH*outW)
        neuron = 0
        for i in range(0, H-PH+1, stride):
            for j in range(0, W-PW+1, stride):
                pool_region = x[index,:,i:i+PH,j:j+PW].reshape(C,PH*PW)
                max_pool_indices = pool_region.argmax(axis=1)
                dout_cur = dout_row[:,neuron]
                neuron += 1
                # pass gradient only through indices of max pool
                dmax_pool = np.zeros(pool_region.shape)
                dmax_pool[np.arange(C),max_pool_indices] = dout_cur
                dx[index,:,i:i+PH,j:j+PW] += dmax_pool.reshape(C,PH,PW)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape

    # Reshape to (N*H*W, C)
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)

    # Use the batch normalization implementation from before
    out, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    # Reshape the output back to (N, H, W, C) and transpose to (N, C, H, W)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # Reshape dout to match the shape used in the forward pass
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)

    # Call batchnorm_backward function to compute gradients
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)

    # Reshape dx_flat back to the original shape
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta




def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
