function [nn, L]  = nntrain_MI(nn, train_x, train_y, opts, val_x, val_y, layer_num)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
%assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
loss.val.min_e             = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

if size(val_y,2) ~= 1
    idx_weak = find(val_y(:,2)==1);
end
m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = floor(m / batchsize);

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');
nn.epochErrL = zeros(numepochs,1);
nn.minErrL = zeros(numepochs,1);
nn.percentErr_Full = zeros(numepochs,1);
nn.percentErr_Minor = zeros(numepochs,1);
L = zeros(numepochs*numbatches,1);
n = 1;
if opts.validation == 1
   er_for_epoch = zeros(numepochs,1); 
    if size(val_y,2) ~= 1
     er_for_epoch_weak = zeros(size(idx_weak,1),1);
    end
end
nn.MI_epoch_XT = zeros(numepochs,1);
nn.MI_epoch_XY = zeros(numepochs,1);

for i = 1 : numepochs
    tic;
    
    if i== 1 && nn.frequency_dependent_layer == 1 
        nn.frequency_dependent_layer =1;
        nn.save_flag = 1;
    elseif i==1
        nn.save_flag =0;
    end
    m = size(train_x, 1);
    kk = randperm(m);
    nn = nnff(nn, train_x, train_y);
    
    
    [ITX, ITY] = mutual_information(nn.a{layer_num+1}(:,2:end), train_y);
    nn.MI_epoch_XT(i) = ITX;
    nn.MI_epoch_XY(i) = ITY;
    
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;

    if opts.validation == 1 %&& i == numepochs
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
        nn.epochErrL(i) = loss.val.e(end); 
        %nn.minErrL(i) = loss.val.min_e(end);
        labels = nnpredict(nn, val_x);
        [dummy, expected] = max(val_y,[],2);
        bad = find(labels ~= expected);    
        er = numel(bad) / size(val_x, 1);
        er_for_epoch(i) = er;
        if size(val_y,2) ~= 1
            str_f_er = sprintf('full_err : %f', er);
            disp(str_f_er);
            nn.percentErr_Full(i) = er;
        end
        
        
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end

if opts.validation == 1 
    nn.epochErr = er_for_epoch;
end
end
