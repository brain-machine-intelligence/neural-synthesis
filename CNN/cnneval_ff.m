function er = cnneval_ff(net, x, y, opts)
    
    net = cnn_fully_ff(net, x, y);
    [dummy, i] = max(net.nn.a{end},[],2);
    labels = i;
    [dummy, expected] = max(y',[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(y, 2);
end
