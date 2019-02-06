%% Bayesian learner for RL

data=[1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1] % simulated or real data here


%1. define beta function
syms q; %q = probability of 1; conversely 1-q is probability of 0
%beta=betapdf([0.01:0.0001:0.99],1.1,1.1); plot(beta); ylim([0 1])
alpha=1; beta=1; beta_f=q^(alpha-1)*(1-q)^(beta-1)*gamma(alpha+beta)/gamma(alpha)/gamma(beta); %fplot(beta, [0 1])
%2. define prior on trial 1 as beta function
prior(1)=beta_f;
%2. for any trial t calculate the updates on Q:
for t=1:length(data)
    t
%LIKELIHOOD for any trial follows a Binomial distribution
%Pr(q|Data)=n!/(k!*(n-k)!)*q^(k)*(1-q)^(n-k)

%k=sum(data(1:t)); %equivalent to how many rewards were recorded on the Left side (note that a No-Reward on the right counts as a reward on the left?)
%sum(reward)
%n=t; %equivalent to how many trials were completed in total
%BERNOILLI: 
k=data(t);

syms q; %q = probability of 1; conversely 1-q is probability of 0
%%%LIKELIHOOD BINOMIAL: likelihood(t+1)=factorial(n)/(factorial(k)*factorial(n-k))*(q^k)*((1-q)^(n-k));
%%%LIKELIHOOD BERNOULLI: 
likelihood(t+1)=(q^k)*((1-q)^(1-k));

%posterior at time t+1 = H*posterior(t)+(1-H)*beta
syms H 
posterior(t)=likelihood(t+1)*prior(t);
nconstant=int(posterior(t),0,1);
posterior(t)=posterior(t)/nconstant;

prior(t+1)=(1-H)*posterior(t)+H*beta_f; % this is the actual posterior of the trial t
prior(t+1)=simplify(prior(t+1));

%exp_q_analytic(t)=int(prior(t+1)*q, q, 0,1);
%exp_H_analytic(t)=int(prior(t+1)*H, H, 0,0.2);

%getting the q and H estimate from the posterior distributions
%EXPECTED VALUE 
%post_func=prior(t+1);
%post_function=matlabFunction(post_func, 'vars', {q, H});
%a=0.01:0.01:0.99;
%b=0.01:0.01:0.2;
%[qq, HH]=ndgrid(a,b);
%Q as expected value based on the posterior:
%for j=1:length(a)
%    for k=1:length(b)
%    post_mat(j,k)=post_function(a(j), b(k));
%    end
%end

end


figure; %fplot(exp_H_analytic(1:10), [0 1])
trial=[4 10 15 20 25 30]
for i=1:6
    ttrial=trial(i)
    ha(i)=subplot(3,2,i)
    ezsurfc(prior(ttrial), [0 .3], [0 1]); title(strcat(num2str(ttrial), 'th Trial')); colormap jet;
    hold on
end

hlink = linkprop(ha,{'CameraPosition','CameraUpVector'});
rotate3d on
