function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case {'softmax','linear'}
            d{n} = - nn.e;
    end
    for i = (n - 1) : -1 : 1
        % Derivative of the activation function
        
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
            case 'ReLU'
                d_act = nn.a{i};
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        if (nn.batch_normalize == 1 && i < (n-1))
            eps = 0.00001;
            N = size(nn.a{i},1);
            if i+1==n % in this case in d{n} there is not the bias term to be removed             
                d_out = d{i + 1}*d_act;
            else
                d_out = d{i + 1}(:,2:end);
            end
            
            dout = d_out;
            dbeta = sum(dout,1);
            dgammax = dout;
            
            dgamma = sum(dgammax.*nn.xhat{i}, 1);
            dxhat = dgammax .* repmat(nn.gamma{i},N,1);
            
            divar = sum(dxhat.* nn.xmu{i},1);
            dxmu1 = dxhat .* repmat(nn.ivar{i},N,1);
            
            dsqrtvar = -1 ./ (nn.sqrtvar{i}.^2) .* divar;
            
            dvar = 0.5 * 1./sqrt(nn.var{i} + eps) .* dsqrtvar;
            
            %%%%%%
            dsq = 1/N * repmat(dvar,N,1);
            %dsq = repmat(dvar,N,1);
            dxmu2 = 2 .* nn.xmu{i} .* dsq;
            
            dx1 = (dxmu1 + dxmu2);
            dmu = -1 .* sum(dxmu1+dxmu2,1);
            %%%%%%%
            
            dx2 = 1/N * repmat(dmu,N,1);
            %dx2 = repmat(dmu,N,1);
            nn.dBa{i+1} = dx1+dx2;
            nn.dgamma{i+1} = dgamma;
            nn.dbeta{i+1} = dbeta;
            
            % Backpropagate first derivatives
            %d{i} = (d{i + 1}(:,2:end) .* nn.dBa{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
            d{i} = (nn.dBa{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
            
        elseif (nn.frequency_dependent_layer == 1)
            if i+1==n
                d{i} = zeros(size(d{i+1},1), size(nn.W{i},2));
                if size(nn.W{i},1) == 1 
                    for kk = 1 : size(d{i+1},1)
                        d{i}(kk,:) = (d{i + 1}(kk,:) * (squeeze(nn.masking{i}(kk,:)).*nn.W{i}) + sparsityError) .* d_act(kk,:);
                    end
                else
                    for kk = 1 : size(d{i+1},1)
                        d{i}(kk,:) = (d{i + 1}(kk,:) * (squeeze(nn.masking{i}(kk,:,:))'.*nn.W{i}) + sparsityError) .* d_act(kk,:);
                    end
                end
            else
                d{i} = zeros(size(d{i+1}(:,2:end),1), size(nn.W{i},2));
                for kk = 1 : size(d{i+1}(:,2:end),1)
                    d{i}(kk,:) = (d{i + 1}(kk,2:end) * (squeeze(nn.masking{i}(kk,:,:))'.*nn.W{i}) + sparsityError) .* d_act(kk,:);
                end
            end
        else
            % Backpropagate first derivatives
            if i+1==n % in this case in d{n} there is not the bias term to be removed             
                d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
            else % in this case in d{i} the bias term has to be removed
                d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
            end
        end
        
        if(nn.dropoutFraction>0 && i > 1)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end
    
    if (nn.batch_normalize == 1)
        i = n-1;
        nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        
        for i = 1 : (n - 2)
           %nn.dW{i} = ((d{i + 1}(:,2:end) .* nn.dBa{i + 1})' * nn.a{i}) / size(nn.dBa{i + 1}, 1);
           nn.dW{i} = (nn.dBa{i + 1}' * nn.a{i}) / size(nn.dBa{i + 1}, 1);
           %nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);
        end
    elseif (nn.frequency_dependent_layer == 1)
        for i = 1 : (n - 1)
            if i+1==n
                nn.dW{i} = zeros(size(d{i+1}',1), size(nn.a{i},2));
                if size(d{i+1}',1) == 1 
                    for kk = 1 : size(d{i+1}',2)
                        nn.dW{i} = nn.dW{i} + squeeze(nn.masking{i}(kk,:)).*(d{i + 1}(kk,:)' * nn.a{i}(kk,:)) / size(d{i + 1}, 1);     
                    end
                else
                    for kk = 1 : size(d{i+1}',2)
                        nn.dW{i} = nn.dW{i} + squeeze(nn.masking{i}(kk,:,:))'.*(d{i + 1}(kk,:)' * nn.a{i}(kk,:)) / size(d{i + 1}, 1);     
                    end
                end
            else
                nn.dW{i} = zeros(size(d{i+1}(:,2:end)',1), size(nn.a{i},2));
                for kk = 1 : size(d{i+1}(:,2:end),1)
                    nn.dW{i} = nn.dW{i} + squeeze(nn.masking{i}(kk,:,:))'.*(d{i + 1}(kk,2:end)' * nn.a{i}(kk,:)) / size(d{i + 1}, 1);     
                end
            end
%             if i==1
%                 nn.dW{i} = zeros(size(d{i+1}(:,2:end)',1), size(nn.a{i},2));
%                  for kk = 1 : size(d{i+1}(:,2:end),1)
%                      nn.dW{i} = nn.dW{i} + squeeze(nn.masking{i}(kk,:,:))'.*(d{i + 1}(kk,2:end)' * nn.a{i}(kk,:)) / size(d{i + 1}, 1);     
%                  end
%             else
%                 if i+1==n
%                     nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);     
%                 else
%                     nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);     
%     
%                 end
%             end
        end
        
       
        
    else
        for i = 1 : (n - 1)
            if i+1==n
                nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);     
            else
                nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);     
    
            end
        end
    end
    
    
end
