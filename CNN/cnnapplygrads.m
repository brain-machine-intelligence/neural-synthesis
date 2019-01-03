function net = cnnapplygrads(net, opts)
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                end
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            end
        end
    end

    net.nn.ffW = net.nn.ffW - net.nn.learningRate * net.dffW;
    
    if net.nn.batch_normalize == 1 
        net.nn.gamma{1} = net.nn.gamma{1} - opts.beta*net.nn.dgamma{1};
        net.nn.beta{1} = net.nn.beta{1} - opts.beta*net.nn.dbeta{1};
    end
    
    for i = 1 : (net.nn.n - 1)
        if(net.nn.weightPenaltyL2>0)
            dW = net.nn.dW{i} + net.nn.weightPenaltyL2 * [zeros(size(net.nn.W{i},1),1) net.nn.W{i}(:,2:end)];
        else
            dW = net.nn.dW{i};
        end
        
        dW = net.nn.learningRate * dW;
        
        %if(net.nn.momentum>0)
        %    net.nn.vW{i} = net.nn.momentum*net.nn.vW{i} + dW;
        %    dW = net.nn.vW{i};
        %end
            
        net.nn.W{i} = net.nn.W{i} - dW;
        
        if net.nn.batch_normalize == 1 && i < (net.nn.n-1)
            net.nn.gamma{i} = net.nn.gamma{i} - opts.beta*net.nn.dgamma{i};
            net.nn.beta{i} = net.nn.beta{i} - opts.beta*net.nn.dbeta{i};
            
         end
    end
    
    
end
