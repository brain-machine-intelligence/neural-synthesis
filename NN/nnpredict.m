function labels = nnpredict(nn, x)
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    if size(nn.a{end},2) == 1
        labels = nn.a{end};
    else
        [dummy, i] = max(nn.a{end},[],2); 
        labels = i; 
    end
end
