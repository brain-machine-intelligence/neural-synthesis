function net = cnnff(net, x,y)
    n = numel(net.layers);
    inputmaps = 3;

    for i = 1: inputmaps
        net.layers{1}.a{i} = squeeze(x(:,:,1,:));
    end
    %net.layers{1}.a{1} = x;
    
    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    
    %  concatenate all end layer feature maps into vector
    net.nn.fv = [];
        
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.nn.fv = [net.nn.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    net.nn.fv = net.nn.fv';
    
    net.nn.fv = [ones(size(net.nn.fv,1),1) net.nn.fv];
    %  feedforward into output perceptrons
    if (net.nn.batch_normalize == 1)
        eps = 0.00001;
        prev_out = net.nn.fv * net.nn.ffW';
        
        fc_N = size(prev_out,1);
        if(net.nn.testing)
            xbar = (prev_out - repmat(net.nn.running_mean{1},fc_N,1)) ./ repmat(sqrt(net.nn.running_var{1} + eps),fc_N,1);
            activations = (repmat(net.nn.gamma{1},fc_N,1) .* xbar) + repmat(net.nn.beta{1},fc_N,1);
        else
            
            mu = (1/fc_N)*sum(prev_out,1);
                
            net.nn.xmu{1} = prev_out - repmat(mu,fc_N,1);
            
            sq = net.nn.xmu{1} .^ 2;
                
            net.nn.var{1} = (1/fc_N) * sum(sq,1);
                
            net.nn.sqrtvar{1} = sqrt(net.nn.var{1} + eps);
                
            net.nn.ivar{1} = 1./net.nn.sqrtvar{1};
                
            net.nn.xhat{1} = net.nn.xmu{1} .* repmat(net.nn.ivar{1},fc_N,1);
            gammax = repmat(net.nn.gamma{1},fc_N,1) .* net.nn.xhat{1};
                
            activations = gammax + repmat(net.nn.beta{1},fc_N,1);
            net.nn.actvations{1} = activations;
            net.nn.running_mean{1} = (net.nn.RAmomentum .* net.nn.running_mean{1}) + (1.0 - net.nn.RAmomentum) .* mu;
            net.nn.running_var{1} =  (net.nn.RAmomentum .* net.nn.running_var{1}) + (1.0 - net.nn.RAmomentum) .* net.nn.var{1};
        end
        
        net.nn.a{1} = sigm(activations);
        net.nn.o = net.nn.a{1};
        net.nn.a{1} = [ones(fc_N,1) net.nn.a{1}];
    else
        net.nn.o = tanh_opt(net.nn.ffW * net.nn.fv + repmat(net.nn.ffb, 1, size(net.nn.fv, 2)));
        
        net.nn.o = net.nn.o';
        fc_m = size(net.nn.o, 1);

        fc_input  = [ones(fc_m,1) net.nn.o];
        net.nn.a{1} = fc_input;
    end
    
    fc_n = net.nn.n;
    for i = 2 : fc_n-1
        if(net.nn.batch_normalize ==1)
            fc_eps = 0.00001;
            prev_out = net.nn.a{i-1}*net.nn.W{i-1}';
            fc_N = size(prev_out,1);
            if(net.nn.testing)
                xbar = (prev_out - repmat(net.nn.running_mean{i},fc_N,1)) ./ repmat(sqrt(net.nn.running_var{i} + fc_eps),fc_N,1);
                activations = (repmat(net.nn.gamma{i},fc_N,1) .* xbar) + repmat(net.nn.beta{i},fc_N,1);
            else
                mu = (1/fc_N)*sum(prev_out,1);
                
                net.nn.xmu{i} = prev_out - repmat(mu,fc_N,1);
            
                sq = net.nn.xmu{i} .^ 2;
                
                net.nn.var{i} = (1/fc_N) * sum(sq,1);
                
                net.nn.sqrtvar{i} = sqrt(net.nn.var{i} + eps);
                
                net.nn.ivar{i} = 1./net.nn.sqrtvar{i};
                
                net.nn.xhat{i} = net.nn.xmu{i} .* repmat(net.nn.ivar{i},fc_N,1);
                gammax = repmat(net.nn.gamma{i},fc_N,1) .* net.nn.xhat{i};
                
                activations = gammax + repmat(net.nn.beta{i},fc_N,1);
                net.nn.actvations{i} = activations;
                net.nn.running_mean{i} = (net.nn.RAmomentum .* net.nn.running_mean{i}) + (1.0 - net.nn.RAmomentum) .* mu;
                net.nn.running_var{i} =  (net.nn.RAmomentum .* net.nn.running_var{i}) + (1.0 - net.nn.RAmomentum) .* net.nn.var{i};
            end
            
            switch net.nn.activation_function
                case 'sigm'
                    net.nn.a{i} = sigm(activations);
                case 'tanh_opt'
                    net.nn.a{i} = tanh_opt(activations);
                case 'ReLU'
                    net.nn.a{i} = ReLU(activations);
                    
            end
        else
            switch net.nn.activation_function
                case 'sigm'
                    net.nn.a{i} = sigm(activations);
                case 'tanh_opt'
                    net.nn.a{i} = tanh_opt(activations);
                case 'ReLU'
                    net.nn.a{i} = ReLU(activations);
            end
            
        end
        
        if(net.nn.dropoutFraction > 0)
            if (net.nn.testing)
                net.nn.a{i} = net.nn.a{i}.*(1 - net.nn.dropoutFraction);
            else
                net.nn.dropOutMask{i} = (rand(size(net.nn.a{i})) > net.nn.dropoutFraction);
                net.nn.a{i} = net.nn.a{i}.*net.nn.dropOutMask{i};
            end
        end
        
        if(net.nn.nonSparsityPenalty>0)
            net.nn.p{i} = 0.99 * net.nn.p{i} + 0.01 * mean(net.nn.a{i}, 1);
        end
        
        net.nn.a{i} = [ones(fc_N,1) net.nn.a{i}];
    end
    
    switch net.nn.output
        case 'sigm'
            net.nn.a{fc_n} = sigm(net.nn.a{fc_n-1} * net.nn.W{fc_n-1}');
        case 'linear'
            net.nn.a{fc_n} = net.nn.a{fc_n-1} * net.nn.W{fc_n-1}';
        case 'softmax'
            net.nn.a{fc_n} = net.nn.a{fc_n-1} * net.nn.W{fc_n-1}';
            net.nn.a{fc_n} = exp(bsxfun(@minus, net.nn.a{fc_n}, max(net.nn.a{fc_n},[],2)));
            net.nn.a{fc_n} = bsxfun(@rdivide, net.nn.a{fc_n}, sum(net.nn.a{fc_n}, 2)); 
    end
    
    net.nn.e = y' - net.nn.a{fc_n};
    net.fin_out = net.nn.a{fc_n};
    switch net.nn.output
        case {'sigm', 'linear'}
            net.nn.L = 1/2*sum(sum(net.nn.e .^2)) / fc_N;
            net.nn.gaError = max(net.nn.e.^2);
        case 'softmax'
            
            net.nn.L = -sum(sum(y.*log(net.nn.a{fc_n}))) / fc_N;    
    end
    
end
