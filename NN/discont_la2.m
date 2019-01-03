function nn = discont_la2(nn, x, y,pop_size, range, iter_num, n_region)
% Train nn.masking for discontinuity resolver
% Add batch normalze by Chang

    n = nn.n;
    m = size(x, 1);
    
    %feedforward pass
    for i = 1 : n-2
       
        input = x;
        output = y;
        
        population = gen_pop_filter(nn.filter{i},pop_size,range);
        for k = 1: iter_num
            pop_size = size(population,1);
            cost = zeros(pop_size,1);
            pop_size_r = size(population);
            pop_size_r = pop_size_r(2:end);
            parfor j = 1: pop_size 
                filter = reshape(population(j,:,:,:), pop_size_r);
                cost(j) = discont_cost2(filter, input, output, nn, i);
            end
            if k==iter_num
                break
            end
            
            [X Y] = sort(cost,'descend');
            
            highest_idx = Y(1: max(floor(size(cost,1) * (0.6 - 0.1*k/iter_num)),3));
            child_population = gen_child_filter(population(highest_idx(:),:,:,:), k, iter_num);
            population(Y(end:-1:end-size(child_population,1)+1),:,:,:) = child_population;
        end
        
        [X Y] = max(cost);
        disp(X)
        
        filter = reshape(population(Y,:,:,:), [size(population,2),size(population,3),size(population,4)]);
        nn.filter{i} = filter;
        
        
    end
        
        
  
end
