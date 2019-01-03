function net = cnntrain(net, x, y, test_x, test_y, opts)
    m = size(x, 4);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.epoch_err = [];
    net.rL = [];
    for i = 1 : opts.numepochs
        net.nn.learningRate = net.nn.learningRate*0.95;
        opts.alpha = opts.alpha*0.95;
        opts.beta = opts.beta*0.95;
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        disp(['lr ' num2str(net.nn.learningRate)]);
        disp(['dr ' num2str(net.nn.dropoutFraction)]);
        net.nn.dropoutFraction = net.nn.dropoutFraction*0.95;
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = cnnff(net, batch_x, batch_y);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.nn.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.nn.L;
        end
        save_drop_out = net.nn.dropoutFraction;
        net.nn.dropoutFraction = 0;
        error_ro = cnneval(net, test_x, test_y,opts)
        net.nn.dropoutFraction = save_drop_out;
        net.epoch_err = [net.epoch_err error_ro];
        toc;
        net.rL(end);
    end
    
    
end
