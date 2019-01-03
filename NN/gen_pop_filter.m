function gen_filter = gen_pop_filter(filter, n, range)
% generate population of new filter
%

    gen_filter = zeros([n+1,size(filter)]);
    tmp_filt = zeros(size(filter));
    for i = 1 :n
        tmp_filt(:,:,1) = filter(:,:,1) + (rand(size(filter(:,:,1)))-0.5)*range;
        tmp_filt(:,:,2) = filter(:,:,2) + (rand(size(filter(:,:,2)))-0.5)*range;
        
        %over_idx = tmp_filt<= -1;
        %tmp_filt(over_idx) = -1;
        over_idx = tmp_filt<= -1;
        tmp_filt(over_idx) = -1;
        over_idx = tmp_filt >= 1;
        tmp_filt(over_idx) = 1;
        
%         [over_idx_x, over_idx_y] = find(tmp_filt(:,:,2) <= -1);
%         tmp_filt(over_idx_x(:),over_idx_y(:),2) = -1;
%         [over_idx_x, over_idx_y] = find(tmp_filt(:,:,2) >= 1);
%         tmp_filt(over_idx_x(:),over_idx_y(:),2) = 1;
        
        %[over_idx_x, over_idx_y] = find(tmp_filt(:,:,1) <= -1);
        %tmp_filt(over_idx_x(:),over_idx_y(:),1) = -1;
        %[over_idx_x, over_idx_y] = find(tmp_filt(:,:,1) >= 1);
        %tmp_filt(over_idx_x(:),over_idx_y(:),1) = 1;
%         
%         max = 1 - tmp_filt(:,:,1) - filter(:,:,2);
%         min = -filter(:,:,2);
%         
%         delta_alpha = ((max-min).*rand(size(filter(:,:,2))) + min)*range;
%         
%         tmp_filt(:,:,2) = filter(:,:,2) + delta_alpha;
%         
%         tmp_filt(over_idx_x(:),over_idx_y(:),2) =0;
        gen_filter(i,:,:,:) = tmp_filt; 
    end
    
    gen_filter(n+1,:,:,:) = filter;


end
