function net = cnn_fully_applygrads(net, opts)
   net.nn.ffW = net.nn.ffW - net.nn.learningRate * net.dffW;
    
    if net.nn.batch_normalize == 1 
        net.nn.gamma{1} = net.nn.gamma{1} - 0.2*net.nn.dgamma{1};
        net.nn.beta{1} = net.nn.beta{1} - 0.2*net.nn.dbeta{1};
    end
    
    
    for i = 1 : (net.nn.n - 1)
        if(net.nn.weightPenaltyL2>0)
            dW = net.nn.dW{i} + net.nn.weightPenaltyL2 * [zeros(size(net.nn.W{i},1),1) net.nn.W{i}(:,2:end)];
        else
            dW = net.nn.dW{i};
        end
        
        dW = net.nn.learningRate * dW;
        
        if(net.nn.momentum>0)
            net.nn.vW{i} = net.nn.momentum*net.nn.vW{i} + dW;
            dW = net.nn.vW{i};
        end
            
        net.nn.W{i} = net.nn.W{i} - dW;
        
        if net.nn.batch_normalize == 1 && i < (net.nn.n-1)
            net.nn.gamma{i} = net.nn.gamma{i} - 0.2*net.nn.dgamma{i};
            net.nn.beta{i} = net.nn.beta{i} - 0.2*net.nn.dbeta{i};
            
         end
    end
    
    
end
