function net = cnnsetup(net, x, y, FC_layer)
    inputmaps = size(x,3);
    
    mapsize = size(squeeze(x(:,:,1)));
    
    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's')
            mapsize = squeeze(mapsize) / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    size_f = size(FC_layer);
    
    fvnum = prod(mapsize) * inputmaps;
    %onum = size(y, 1);
    onum = FC_layer(1);
    net.nn.ffb = zeros(onum, 1);
    net.nn.ffW = (rand(onum, fvnum+1) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
    
    net.nn.size   = FC_layer;
    net.nn.n      = numel(net.nn.size);
    
    net.nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    net.nn.learningRate                     = 0.6;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    net.nn.momentum                         = 0.5;          %  Momentum
    net.nn.RAmomentum                       = 0.8;
    net.nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
    net.nn.weightPenaltyL2                  = 0;            %  L2 regularization
    net.nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    net.nn.sparsityTarget                   = 0.05;         %  Sparsity target
    net.nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    net.nn.dropoutFraction                  = 0.0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    net.nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    net.nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    
    net.nn.batch_normalize                  = 1;
    net.nn.frequency_dependent_layer        = 0;
    net.nn.batchsize                        = 10;
    net.first_ff                            = 1;
    
    
    net.nn.gamma{1}                         =  ones(1,net.nn.size(1));
    net.nn.beta{1}                          =  ones(1,net.nn.size(1));
    	
    net.nn.running_mean{1}                  = rand(1, net.nn.size(1));
    net.nn.running_var{1}                   = rand(1, net.nn.size(1));
    
    for i = 2 : net.nn.n   
        % weights and weight momentum
        net.nn.W{i - 1} = (rand(net.nn.size(i), net.nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (net.nn.size(i) + net.nn.size(i - 1)));
        net.nn.vW{i - 1} = zeros(size(net.nn.W{i - 1}));
        
        net.nn.gamma{i} = ones(1,net.nn.size(i));
        net.nn.beta{i} = ones(1,net.nn.size(i));
        net.nn.activations{i-1} = zeros(net.nn.size(i));
        
        net.nn.running_mean{i} = rand(1,net.nn.size(i));
        net.nn.running_var{i} = rand(1,net.nn.size(i));
        % average activations (for use with sparsity)
        net.nn.p{i}     = zeros(1, net.nn.size(i));
        
        %nn.filter{i-1} = (rand(nn.size(i-1)+1, nn.size(i),2)-0.5)*2;
        
        net.nn.filter{i-1} = zeros(net.nn.size(i-1)+1, net.nn.size(i),2);
        %nn.filter{i-1}(:,:,1) = -1;
        net.nn.filter{i-1}(:,:,1) = -1;
        net.nn.filter{i-1}(:,:,2) = 1;
        
        
        %nn.filter{i-1}(:,:,1) = (rand(nn.size(i-1)+1, nn.size(i),1)-0.5)*2;
        %nn.filter{i-1}(:,:,2) = rand(nn.size(i-1)+1, nn.size(i),1).*(1- nn.filter{i-1}(:,:,1));
        
    end
    
end
