function er = cnneval(net, x, y, opts)
    
    net.nn.testing = 1;
    net = cnnff(net, x, y);
    net.nn.testing = 0;
    [dummy, i] = max(net.nn.a{end},[],2);
    labels = i;
    [dummy, expected] = max(y',[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(y, 2);
end
