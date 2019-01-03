function L = discont_cost2(filter, x, y, nn,layer_num)
% Train nn.masking for discontinuity resolver
%
    save_filter = nn.filter{layer_num};
    nn.filter{layer_num} = filter;
    
    nn = nnff(nn, x, y);
    
    nn.filter{layer_num} = save_filter;
    L = -nn.L;
    %abs_e = abs(nn.e);
    
    
    %mean_sq_err = sum((nn.a{1,nn.n} - nn.L).^2);
    %err = (nn.a{1,nn.n} -y).^2;
    %mean_sq_err = min(err)/std(err);
%     combination = nchoosek(1:size(nn.e,1),2);
     %nn.filter{layer_num} = save_filter;
%     x_diff = abs(x(combination(:,1),:) - x(combination(:,2),:));
%     squared_sum = sum(x_diff,2)+0.0001;
%     abs_diff = abs(abs_e(combination(:,1),:) - abs_e(combination(:,2),:));
%     sum_diff = sum(abs_diff,2)+0.0001;
%     loss_all = sum_diff ./squared_sum;
%     L=-sum(loss_all,1);
    %L = -mean_sq_err;
end
