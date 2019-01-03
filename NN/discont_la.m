function nn = discont_la(nn, x, y,pop_size, range, iter_num)
% Train nn.masking for discontinuity resolver
% Add batch normalze by Chang

    n = nn.n;
    m = size(x, 1);
    
    %feedforward pass
    for i = n-1 :-1: 1
       
        input = nn.a{i};
        output = y;
        %first_cost = discont_cost(nn.filter{1}, nn.a{1},output, nn, 1 );
        population = gen_pop_filter(nn.filter{i},pop_size,range);
        for k = 1: iter_num
            pop_size = size(population,1);
            cost = zeros(pop_size,1);
            for j = 1: pop_size 
                filter = reshape(population(j,:,:,:), [size(population,2),size(population,3),size(population,4)]);
                cost(j) = discont_cost(filter, input, output, nn, i);
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
     
   % final_cost = discont_cost(nn.filter{1}, nn.a{1},output, nn, 1 );
        
        
  
end
