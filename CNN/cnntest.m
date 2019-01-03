function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net.nn.testing = 1;
    net = cnnff(net, x, y);
    net.nn.testing = 0;
    [~, h] = max((net.fin_out)');
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
    er
end
