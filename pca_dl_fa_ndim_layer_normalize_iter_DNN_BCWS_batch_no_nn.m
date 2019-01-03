% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% -------------------------------------------------------- %
% Firefly Algorithm for constrained optimization using     %
% for the design of a spring (benchmark)                   % 
% by Xin-She Yang (Cambridge University) Copyright @2009   %
% -------------------------------------------------------- %

function [bestsolution, x, bestojb, fitplot] = pca_dl_fa_ndim_layer_normalize_iter_DNN_batch_no_nn(x0,name,layer, dim, save_fa_data,nn)
% parameters [n N_iteration alpha betamin gamma]
%para=[100 250 0.8 0.8 1];
para=[100 250 0.5 0.2 1];
%help fa_ndim.m

% Simple bounds/limits for d-dimensional problems
d=dim;
load(save_fa_data);
ta = mean(x0);
thresh = ta*d*nn.range;
Lb=-thresh*ones(1,d);
Ub=thresh*ones(1,d);
%disp(['bound : ' num2str(thresh)]);
% Initial random guess
u0=Lb+(Ub-Lb).*rand(1,d);

[u,fval,NumEval,~, fit]=ffa_mincon(@cost,u0,Lb,Ub,para, x0,layer, save_fa_data, nn);

% Display results


bestsolution=u;
bestojb=fval;
total_number_of_function_evaluations=NumEval;
fitplot = fit;
x = x0 + (bestsolution * v(idx(1:d),:));
x = normalizeHak(x);
%x = normalize(x, muval, sigmaval);
%x= x0 + (nbest(1,:) * v(:,idx(1:length2))');
%filename = sprintf('plot_result_%d.png',name);
%fname = strcat(nn.filename,filename);
%fn = sprintf('%sres2_%s', nn.filename, filename); 

%xlabel('Generation','fontsize',17,'fontname','arial');
%ylabel('Fitness Value','fontsize',17,'fontname','arial');
%legend_h=legend('x0 - x','DL(x0) - DL(x)');
%set(legend_h,'FontSize',14);
%saveas(gcf,fn);

%%% Put your own cost/objective function here --------%%%
%% Cost or Objective function
 function z=cost(w,v,nn,x0,idx,sol,layer)
% Exact solutions should be (1,1,...,1) 

[dum, length] = size(w(1,:));
x= x0 + (w(1,:) * v(idx(1:length),:));
x = normalizeHak(x);
%x = normalize(x, muval, sigmaval);
A = x0-x;
%fit1 = 1/corr(x0',x');
fit1 = norm(A);

m = size(x, 1);


tmp_y0 = [0 0];
tmp_y = [0 0];
tmp_x_gr = [x0; x];
tmp_y_gr = [tmp_y0; tmp_y];
nn.testing = 1;
tmp_nn = nnff(nn, tmp_x_gr, tmp_y_gr);
nn.testing = 0;

DLx0 = tmp_nn.a{layer}(1,:);
DLx = tmp_nn.a{layer}(2,:);

A = DLx0 - DLx;
%fit2 = corr(DLx0',DLx');
fit2 = norm(A);
c = nn.cval;
z = fit1 - (c*fit2);



%%% End of the part to be modified -------------------%%%

%%% --------------------------------------------------%%%
%%% Do not modify the following codes unless you want %%%
%%% to improve its performance etc                    %%%
% -------------------------------------------------------
% ===Start of the Firefly Algorithm Implementation ======
%         Lb = lower bounds/limits
%         Ub = upper bounds/limits
%   para == optional (to control the Firefly algorithm)
% Outputs: nbest   = the best solution found so far
%          fbest   = the best objective value
%      NumEval = number of evaluations: n*MaxGeneration
% Optional:
% The alpha can be reduced (as to reduce the randomness)
% ---------------------------------------------------------

% Start FA
function [nbest,fbest,NumEval, x, fit]...
           =ffa_mincon(fhandle,u0, Lb, Ub, para, x0, layer, save_fa_data, nn)
% Check input parameters (otherwise set as default values)
if nargin<5, para=[20 500 0.25 0.20 1]; end
if nargin<4, Ub=[]; end
if nargin<3, Lb=[]; end
if nargin<2,
disp('Usuage: FA_mincon(@cost,u0,Lb,Ub,para)');
end

% n=number of fireflies
% MaxGeneration=number of pseudo time steps
% ------------------------------------------------
% alpha=0.25;      % Randomness 0--1 (highly random)
% betamn=0.20;     % minimum value of beta
% gamma=1;         % Absorption coefficient
% ------------------------------------------------
n=para(1);  MaxGeneration=para(2);
alpha=para(3); betamin=para(4); gamma=para(5);


fit = zeros(1,MaxGeneration);


% Total number of function evaluations
NumEval=n*MaxGeneration;

% Check if the upper bound & lower bound are the same size
if length(Lb) ~=length(Ub),
    disp('Simple bounds/limits are improper!');
    return
end
load(save_fa_data);

% Calcualte dimension
d=length(u0);

% Initial values of an array
zn=ones(n,1)*10^100;
% ------------------------------------------------
% generating the initial locations of n fireflies
[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0);
hold on;
% Iterations or pseudo time marching
%초기화
for num = 1: layer;
    sol.a0{num} = nn.a{num};
    sol.a{num} = nn.a{num};
end
for k=1:MaxGeneration,     %%%%% start iterations

% This line of reducing alpha is optional
 alpha=alpha_new(alpha,MaxGeneration);

% Evaluate new solutions (for all n fireflies)

for i=1:n,
   zn(i)=fhandle(ns(i,:),v,nn,x0,idx,sol,layer);
   Lightn(i)=zn(i);
end

% Ranking fireflies by their light intensity/objectives
[Lightn,Index]=sort(zn,'descend');
ns_tmp=ns;
for i=1:n,
 ns(i,:)=ns_tmp(Index(i),:);
end

%% Find the current best
nso=ns; Lighto=Lightn;
nbest=ns(1,:); Lightbest=Lightn(1);

% For output only
fbest=Lightbest;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[dum, length2] = size(nbest(1,:));
x= x0 + (nbest(1,:) * v(idx(1:length2),:));
x = normalizeHak(x);
%x = normalize(x, muval, sigmaval);

m = size(x, 1);

A = x0-x;
%fit1 = 1/corr(x0',x');
fit1 = norm(A);

%random_noise = rand(size(x0))*0.2;
%x2 = x0 + random_noise;

            
%tmp_y2 = [0 0 0 0 0 0 0 0 0 1];
tmp_y0 = [0 0];
tmp_y = [0 0];
tmp_x_gr = [x0; x;];
tmp_y_gr = [tmp_y0; tmp_y;];
nn.testing = 1;
tmp_nn = nnff(nn, tmp_x_gr, tmp_y_gr);
nn.testing = 0;

%DLx2 = tmp_nn.a{layer}(3,:);
DLx0 = tmp_nn.a{layer}(1,:);
DLx = tmp_nn.a{layer}(2,:);


A = DLx0 - DLx;
%B = DLx0 - DLx2;
%fit2 = corr(DLx0',DLx');
fit2 = (norm(A));
c = nn.cval;
fit2 = c*fit2;
z = fit1 - fit2;
%plot(k,fit1,'go');
%plot(k,fit2,'bx');
%hold on;
fit(1,k) = z;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Move all fireflies to the better locations
[ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
      Lightbest,alpha,betamin,gamma,Lb,Ub);


end   %%%%% end of iterations

% -------------------------------------------------------
% ----- All the subfunctions are listed here ------------
% The initial locations of n fireflies
function [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)
  % if there are bounds/limits,
if length(Lb)>0,
   for i=1:n,
   ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
   end
else
   % generate solutions around the random guess
   for i=1:n,
   ns(i,:)=u0+randn(1,d);
   end
end

% initial value before function evaluations
Lightn=ones(n,1)*10^100;

% Move all fireflies toward brighter ones
function [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,...
             nbest,Lightbest,alpha,betamin,gamma,Lb,Ub)
% Scaling of the system
scale=abs(Ub-Lb);

% Updating fireflies
for i=1:n,
% The attractiveness parameter beta=exp(-gamma*r)
   for j=1:n,
      r=sqrt(sum((ns(i,:)-ns(j,:)).^2));
      % Update moves
if Lightn(i)<Lighto(j), % Brighter and more attractive %< 최대값 찾기 > 최소값 찾기
   beta0=1; beta=(beta0-betamin)*exp(-gamma*r.^2)+betamin;
   tmpf=alpha.*(rand(1,d)-0.5).*scale;
   ns(i,:)=ns(i,:).*(1-beta)+nso(j,:).*beta+tmpf;
end
   end % end for j

end % end for i

% Check if the updated solutions/locations are within limits
[ns]=findlimits(n,ns,Lb,Ub);

% This function is optional, as it is not in the original FA
% The idea to reduce randomness is to increase the convergence,
% however, if you reduce randomness too quickly, then premature
% convergence can occur. So use with care.
function alpha=alpha_new(alpha,NGen)
% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
% alpha_0=0.9
delta=1-(10^(-4)/0.9)^(1/NGen);
alpha=(1-delta)*alpha;

% Make sure the fireflies are within the bounds/limits
function [ns]=findlimits(n,ns,Lb,Ub)
for i=1:n,
     % Apply the lower bound
  ns_tmp=ns(i,:);
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);

  % Apply the upper bounds
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move
  ns(i,:)=ns_tmp;
end

%% ==== End of Firefly Algorithm implementation ======

