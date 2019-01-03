function L = discont_cost(filter, x, y, nn,layer_num)
% Train nn.masking for discontinuity resolver
%
    n= nn.n;
    input =  x ;
    i_filter = filter;
    for i = layer_num : n-1
        
        small = min(i_filter,[],3);
        big = max(i_filter,[],3);

        small_rep = repmat(small, 1,1,size(input,1));
        big_rep = repmat(big, 1,1,size(input,1));
        small_rep = permute(small_rep, [3 1 2]);
        big_rep = permute(big_rep, [3 1 2]);

        prev_out_rep = repmat(input, 1,1,size(i_filter,2));

        masking = (small_rep < prev_out_rep) & (prev_out_rep < big_rep);
        result = zeros(size(input,1), size(nn.W{i},1));
        for kk = 1 : size(input,1)
            if i == n-1 
                result(kk,:) = input(kk,:,:) * (squeeze(masking(kk,:,:)).*nn.W{i})';   
            else
                result(kk,:) = input(kk,:,:) * (squeeze(masking(kk,:,:)).*nn.W{i}');   
                
            end
        end
        if i== n-1 
            activation = nn.output;
        else 
            activation = nn.activation_function;
        end
        
        switch activation
            case 'sigm'
                out = sigm(result);
            case 'linear'
                out = result;
            case 'softmax'
                out = result;
                out = exp(bsxfun(@minus, out, max(out,[],2)));
                out = bsxfun(@rdivide, out, sum(out, 2));
            case 'tanh_opt' 
                out = tanh_opt(result);
            case  'ReLU'
                out = ReLU(result);
        end
        
        if i== n-1 
            break
        else
            m = size(out, 1);
            input = [ones(m,1) out];
            i_filter = nn.filter{i+1};
        end
    end
    
    
    L = -sum(out-y).^2;
end
