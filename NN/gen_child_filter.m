function child_filter = gen_child_filter(parent, epoch, tot_epoch)
% generate population of new filter
%

    tot_size = size(parent,1);
    cross_over_ratio = 0.7 - 0.6*epoch/tot_epoch;
    mutation_ratio = 0.6 - 0.4*epoch/tot_epoch;
    cross_over_part = 0.4 - 0.35*epoch/tot_epoch;
    child_filter = zeros(size(parent));
    %%crossover
    for i = 1: tot_size
        if rand(1) < cross_over_ratio
            parent1_filter = reshape(parent(i,:,:,:), [size(parent,2),size(parent,3),size(parent,4)]);
            parent2_idx = floor(rand(1)*(tot_size+1));
            if parent2_idx ==0 
                parent2_idx = 1;
            elseif parent2_idx == tot_size+1
                parent2_idx = tot_size;
            end
            parent2_filter = reshape(parent(parent2_idx,:,:,:), [size(parent,2),size(parent,3),size(parent,4)]);
            
            idx = rand(size(parent1_filter,1) , size(parent1_filter,2),1) < cross_over_part;
            if size(parent1_filter,2)==1 
                parent1_filter(idx) = parent2_filter(idx);
            else
                parent1_filter(idx) = parent2_filter(idx);
            end
            child_filter(i,:,:,:) = parent1_filter;
        else
            child_filter(i,:,:,:) = parent(i,:,:,:);
        end
    end
    
    %%mutation_s_point
    [idx] = rand(size(child_filter)) < mutation_ratio;
    child_filter(idx) = child_filter(idx) + (rand(size(child_filter(idx)))-0.5)*mutation_ratio;
    
    
    %over_idx = child_filter <= -1;
    %child_filter(over_idx(:)) = -1;
    over_idx = child_filter <= -1;
    child_filter(over_idx(:)) = -1;
    
    over_idx = child_filter >= 1;
    child_filter(over_idx(:)) = 1;
   
%     tmp_filter1 = child_filter(:,:,:,1);
%     [m_x, m_y] = find(((rand(size(tmp_filter1))) < mutation_ratio) ==1);
%     
%     tmp_filter1(m_x(:),m_y(:)) = tmp_filter1(m_x(:),m_y(:)) + (rand(size(m_x,1),size(m_y,1))-0.5)*mutation_ratio;
%     
%     idx = find(tmp_filter1 < -1);
%     tmp_filter1(idx(:)) = -1;
%     
%     idx = find(tmp_filter1 > 1);
%     tmp_filter1(idx(:)) = 1;
%     
%     child_filter(:,:,:,1) = tmp_filter1;
%     %mutation_interval
%     tmp_filter2 = child_filter(:,:,:,2);
%     [m_x, m_y] = find(((rand(size(tmp_filter2))) < mutation_ratio) ==1);
%     
%     max = 1 - tmp_filter1(m_x(:),m_y(:)) - tmp_filter2(m_x(:),m_y(:));
%     min = -tmp_filter2(m_x(:),m_y(:));
%     delta_alpha = ((max-min).*rand(size(tmp_filter2(m_x(:),m_y(:)))) + min)*mutation_ratio;
%     
%     tmp_filter2(m_x(:),m_y(:)) = tmp_filter2(m_x(:),m_y(:)) + delta_alpha;
%     tmp_filter2(idx(:)) = 0;
%     child_filter(:,:,:,2) = tmp_filter2;
    


end
