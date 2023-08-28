from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Reshape the input tensor x into a matrix of shape (N, D)
    x_reshape = np.reshape(x, (x.shape[0], -1))
    # Compute the dot product between x_reshape and w, and add b 
    # (the bias vector b to the result
    out = np.dot(x_reshape, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)
    """ 
    This next line computes the gradient with respect to the weight matrix
    w using the chain rule. It first reshapes the input x into a 2D matrix
    where the first dimension is the number of examples N and the second 
    dimension is the flattened feature vector. Then, it takes the dot product
    of the transpose of the reshaped x and the upstream derivative dout. 
    The resulting matrix has the same shape as w, so it gives us the gradient
    of the loss with respect to the weight matrix w.
    """
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    
    # compute the gradient with respect to the biases b.
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, 0)  # Apply ReLU activation


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    x_pad = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    x_pad = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], mode='constant')

    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Compute the dot product between the receptive field and 
                    # the filter weights w[f], and add the bias term b[f].
                    out[n, f, i, j] = (x_pad[n, :, h_start:h_end, w_start:w_end] * w[f, :, :, :]).sum() + b[f]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_pad, w, b, conv_param = cache
    N, F, outH, outW = dout.shape
    N, C, padH, padW = x_pad.shape
    FH, FW = w.shape[2], w.shape[3]
    stride = conv_param['stride']
    pad = conv_param['pad']

    filter_vec_len = C*FH*FW
    output_vec_len = outH*outW

    # initialize gradients
    dx = np.zeros((N, C, padH - 2*pad, padW - 2*pad))
    dw, db = np.zeros(w.shape), np.zeros(b.shape)

    # w_row stores the relevant w values that we will use
    w_row = w.reshape(F, filter_vec_len)

    # x_col stores the relevant x values that we will use or more 
    # correctly the coresponding derevative from dout
    x_col = np.zeros((filter_vec_len, output_vec_len))

    # Iterate over all data points
    for i in range(N):
        # Get the derivative for the current data point and reshape it
        dout_i = dout[i].reshape(F, output_vec_len)
        
        # Compute the convolution between the current input and the current filter
        # using matrix multiplication and reshape it
        w_out_i = w_row.T.dot(dout_i)
        
        # Initialize the gradient of the input tensor for the current data point
        dx_i = np.zeros((C, padH, padW))
        
        # Keep track of the position in the output tensor
        out_pos = 0
        
        # Iterate over all possible positions for the filter in the input tensor
        for h in range(0, padH - FH + 1, stride):
            for w in range(0, padW - FW + 1, stride):
                # Add the contribution of the current filter position to the gradient
                dx_i[:, h:h+FH, w:w+FW] += w_out_i[:, out_pos].reshape(C, FH, FW)
                
                # Flatten the input tensor patch for the current filter position
                x_col_i = x_pad[i, :, h:h+FH, w:w+FW].reshape(C*FH*FW)
                
                # Store the input tensor patch in the column matrix
                x_col[:, out_pos] = x_col_i
                
                # Move to the next position in the output tensor
                out_pos += 1
        
        # Store the gradient of the input tensor for the current data point
        dx[i] = dx_i[:, pad:-pad, pad:-pad]
        
        # Compute the gradients for the filter and the bias using matrix multiplication
        dw += dout_i.dot(x_col.T).reshape(F, C, FH, FW)
        db += dout_i.sum(axis=1)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    outH = int(1 + (H - pool_height) / stride)
    outW = int(1 + (W - pool_width) / stride)
    
    # initialize output tensor
    out = np.zeros((N, C, outH, outW))

    for i in range(outH):
        for j in range(outW):
            # get the current pooling region
            pool_region = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            # apply max operation over each pooling region
            out[:, :, i, j] = np.max(pool_region, axis=(2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    out_height = 1 + (H - pool_height) // stride
    out_width = 1 + (W - pool_width) // stride
    
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    h_start = h_out * stride
                    w_start = w_out * stride
                    h_end = h_start + pool_height
                    w_end = w_start + pool_width
                    pool_region = x[n, c, h_start:h_end, w_start:w_end]
                    max_indices = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                    dx[n, c, h_start:h_end, w_start:w_end][max_indices] = dout[n, c, h_out, w_out]
           
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


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
