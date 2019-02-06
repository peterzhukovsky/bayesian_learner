%% Bayesian learner for RL

data=[1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0] % simulated or real data here


%1. define beta function
syms q; %q = probability of 1; conversely 1-q is probability of 0
%beta=betapdf([0.01:0.0001:0.99],1.1,1.1); plot(beta); ylim([0 1])
alpha=1; beta=1; beta_f=q^(alpha-1)*(1-q)^(beta-1)*gamma(alpha+beta)/gamma(alpha)/gamma(beta); %fplot(beta, [0 1])
%2. define prior on trial 1 as beta function
prior(1)=beta_f;
%2. for any trial t calculate the updates on Q: q left vs q right separate;
%q left only updated when the agent learns about LEFT; otherwise left the same;
%also q right updated only when the agent learns about RIGHT, so L/R are
%not conjoint. 
for t=1:length(data)
    t
    memorysize=1;
    if t>memorysize;
        data_useful=data(t-memorysize:t);
        k=sum(data_useful); 
    elseif t<=memorysize;
        data_useful=data(1:t);
        k=sum(data_useful); 
    end
    
%sum(reward)
n=length(data_useful); %equivalent to how many trials were completed in total


syms q; %q = probability of 1; conversely 1-q is probability of 0
%LIKELIHOOD for any trial follows a Binomial distribution
%Pr(q|Data)=n!/(k!*(n-k)!)*q^(k)*(1-q)^(n-k)
likelihood(t+1)=factorial(n)/(factorial(k)*factorial(n-k))*(q^k)*((1-q)^(n-k));

likelihood(t+1)=simplify(likelihood(t+1));
%posterior at time t+1 = H*posterior(t)+(1-H)*beta
H=20/25; %%%!!!!!!!!!!!!!!!!! Optimize later!
posterior(t)=likelihood(t+1)*prior(t);
nconstant=int(posterior(t),0,1);
posterior(t)=posterior(t)/nconstant;
posterior(t)=simplify(posterior(t));
prior(t+1)=(1-H)*posterior(t)+H*beta_f; % this is the actual posterior of the trial t

exp_q_analytic(t)=double(int(prior(t+1)*q, 0,1));
%getting the q estimate from each of the distributions
%EXPECTED VALUE vs HIGHEST PROBABILITY --- q=0:0.01:1; expectedqval(t)=trapz(q,post_function(q))*(1/101)
post_func=prior(t+1);
post_function=matlabFunction(post_func, 'vars', {q});
a=0.01:0.01:0.99;

%Q as expected value based on the posterior:
exp_q(t)=sum(post_function(a)*a')/length(a);
%
%integr_post=int(posterior(t));
%integr_post_func=matlabFunction(integr_post, 'vars', {q});
%integral(integr_post_func, 0,1)
end


%Use softmax to calculate the probability of Left vs Right response and the
%joint dataset probability => optimize the parameters to maximize the joing
%probability (or minimize negative log PDF)


%% plotting some of the readouts from the Bayesian learner above

figure; fplot(posterior(13:22), [0 1])
figure; plot(exp_q_analytic); ylim([0 1])
contingency(1:14)=0.85;
contingency(15:30)=0.15;
hold on
plot(contingency, 'r'); hold off;

