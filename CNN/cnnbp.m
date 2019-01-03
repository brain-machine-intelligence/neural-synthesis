function net = cnnbp(net, y)
    fc_n = net.nn.n;
    sparsityError = 0;
    switch net.nn.output
        case 'sigm'
            d{fc_n} = -net.nn.e .* (net.nn.a{fc_n} .* (1 - net.nn.a{fc_n}));
        case {'softmax','linear'}
            d{fc_n} = -net.nn.e;
    end
    
    for i = (fc_n-1) : -1 : 1
        switch net.nn.activation_function
            case 'sigm'
                d_act = net.nn.a{i} .* (1- net.nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1- 1/(1.7159)^2 * net.nn.a{i}.^2);
            case 'ReLU'
                d_act = net.nn.a{i};
        end
        
        if(net.nn.nonSparsityPenalty>0)
            pi = repmat(net.nn.p{i}, size(net.nn.a{i}, 1), 1);
            sparsityError = [zeros(size(net.nn.a{i},1),1) net.nn.nonSparsityPenalty * (-net.nn.sparsityTarget ./ pi + (1 - net.nn.sparsityTarget) ./ (1 - pi))];
        end
        
        if (net.nn.batch_normalize == 1 && i < (fc_n-1))
            eps = 0.00001;
            fc_N = size(net.nn.a{i},1);
            if( i+1 == fc_n)
                d_out = d{i+1}*d_act;
            else
                d_out = d{i+1}(:,2:end);
            end
            
            dout = d_out;
            dbeta = sum(dout,1);
            dgammax = dout;
            
            dgamma = sum(dgammax.*net.nn.xhat{i+1},1);
            dxhat = dgammax.*repmat(net.nn.gamma{i+1}, fc_N,1);
            
            divar = sum(dxhat .* net.nn.xmu{i+1},1);
            dxmu1 = dxhat .* repmat(net.nn.ivar{i+1},fc_N,1);
            
            dsqrtvar = -1 ./ (net.nn.sqrtvar{i+1}.^2) .* divar;
            dvar = 0.5 * 1./sqrt(net.nn.var{i+1} + eps) .* dsqrtvar;
            
            dsq = 1/fc_N * repmat(dvar, fc_N,1);
            
            dxmu2 = 2 .* net.nn.xmu{i+1} .* dsq;
            
            dx1 = (dxmu1 + dxmu2);
            dmu = -1 .* sum(dxmu1 + dxmu2,1);
            
            dx2 = 1/fc_N * repmat(dmu, fc_N,1);
            
            net.nn.dBa{i+1} = dx1+dx2;
            net.nn.dgamma{i+1} = dgamma;
            net.nn.dbeta{i+1} = dbeta;
            
            d{i} = (net.nn.dBa{i+1} * net.nn.W{i} + sparsityError) .* d_act;
        else
            if i+1==fc_n
                d{i} = (d{i+1} * net.nn.W{i} + sparsityError) .* d_act;
            else
                d{i} = (d{i+1}(:,2:end) * net.nn.W{i} + sparsityError) .*d_act;
            end
        end
        
        if(net.nn.dropoutFraction>0 && i >1)
            d{i} = d{i} .* [ones(size(d{i},1),1) net.nn.dropOutMask{i}];
        end
        
    end
    
    
    if(net.nn.batch_normalize ==1)
        i = fc_n-1;
        net.nn.dW{i} = (d{i+1}' * net.nn.a{i}) / size(d{i+1},1);
        
        for i = 1: (fc_n-2);
            net.nn.dW{i} = (net.nn.dBa{i+1}' * net.nn.a{i}) / size(net.nn.dBa{i+1}, 1);
        end
    else
        for i = 1:(fc_n-1)
            if i+1 == fc_n
                net.nn.dW{i} = (d{i+1}' * net.nn.a{i}) / size(d{i+1},1);
            else
                net.nn.dW{i} = (d{i + 1}(:,2:end)' * net.nn.a{i}) / size(d{i + 1}, 1); 
               
            end
        end
    end
    n = numel(net.layers);

    %  error
   % net.nn.e = net.o - y;
    %  loss function
   % net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    
    if (net.nn.batch_normalize == 1)
        %d{i} = (net.nn.dBa{i+1} * net.nn.W{i} + sparsityError) .* (net.nn.o .* (1 - net.nn.o);
        
        %net.od = (net.nn.dBa{2} * net.nn.ffW + sparsityError) .* (net.nn.o .* (1 - net.nn.o));
        net.od = d{1}(:,2:end); %.* (net.nn.o .* (1 - net.nn.o));   %  output delta
        
        eps = 0.00001;
        fc_N = size(net.nn.a{1},1);
        
        d_out = net.od;
            
        dout = d_out;
        dbeta = sum(dout,1);
        dgammax = dout;
            
        dgamma = sum(dgammax.*net.nn.xhat{1},1);
        dxhat = dgammax.*repmat(net.nn.gamma{1}, fc_N,1);
            
        divar = sum(dxhat .* net.nn.xmu{1},1);
        dxmu1 = dxhat .* repmat(net.nn.ivar{1},fc_N,1);
            
        dsqrtvar = -1 ./ (net.nn.sqrtvar{1}.^2) .* divar;
        dvar = 0.5 * 1./sqrt(net.nn.var{1} + eps) .* dsqrtvar;
           
        dsq = 1/fc_N * repmat(dvar, fc_N,1);
            
        dxmu2 = 2 .* net.nn.xmu{1} .* dsq;
            
        dx1 = (dxmu1 + dxmu2);
        dmu = -1 .* sum(dxmu1 + dxmu2,1);
            
        dx2 = 1/fc_N * repmat(dmu, fc_N,1);
            
        net.nn.dBa{1} = dx1+dx2;
        net.nn.dgamma{1} = dgamma;
        net.nn.dbeta{1} = dbeta;
        
        %net.fvd = (net.od * net.nn.ffW);              %  feature vector delta
        
        net.fvd = (net.nn.dBa{1} * net.nn.ffW + sparsityError).* (net.nn.fv .* (1 - net.nn.fv)); % sigmoid
        %net.fvd = (net.nn.dBa{1}.* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * net.nn.o.^2)) * net.nn.ffW + sparsityError);
    else
        net.od = d{1}(:,2:end) .* (net.nn.o .* (1 - net.nn.o));   %  output delta   sigm
        net.fvd = (net.od * net.nn.ffW);              %  feature vector delta
    end
    
    
    
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.nn.fv .* (1 - net.nn.fv));
        
        %net.fvd = net.fvd .* 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * net.nn.fv.^2);  % deriv of tanh
    end
    net.fvd = net.fvd';
    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    
    net.dffW = (net.nn.dBa{1})' * net.nn.fv / size(net.nn.dBa{1}, 1);
   
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
