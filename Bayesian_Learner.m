%%Bayesian learner for RL

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
H=1/25; %%%!!!!!!!!!!!!!!!!! Optimize later!
posterior(t)=likelihood(t+1)*prior(t);
nconstant=int(posterior(t),0,1);
posterior(t)=posterior(t)/nconstant;

prior(t+1)=(1-H)*posterior(t)+H*beta_f; % this is the actual posterior of the trial t
%expected value of q:
exp_q_analytic(t)=int(prior(t+1)*q, 0,1);

end


%% plotting some of the readouts from the Bayesian learner above

figure; fplot(posterior(13:22), [0 1])
figure; plot(exp_q_analytic); ylim([0 1])
contingency(1:14)=0.85;
contingency(15:30)=0.15;
hold on
plot(contingency, 'r'); hold off;



fplot(prior(t+1), [0 1])
hold on
fplot(post_func, [0 1])

fplot(posterior(30), [0 1])


plot(a, post_function(a))



figure
fplot(posterior, [0 1]); 


%%%%%%%%%%%%%%%%%% discrete alterantive
%PDF_FUNCTION = matlabFunction(f, 'vars', {q})
%fplot(PDF_FUNCTION)

q=[0.01:0.01:0.99]
for i=1:length(q)
likelihood(i)=factorial(n)/(factorial(k)*factorial(n-k))*(q(i)^k)*((1-q(i))^(n-k))
end


%Function minimization
%A=[]; b=[]; Aeq=[]; beq=[]; lb=0; ub=1; x0=0.4; q_max=fmincon(post_function,x0,A,b,Aeq,beq,lb,ub)
%fminsearch(post_function, [0.1 0.9])

%getting the q estimate from each of the distributions
post_func=prior(t+1);
post_function=matlabFunction(post_func, 'vars', {q});
a=0.01:0.01:0.99;
exp_q(t,:)=sum(post_function(a)*a')/length(a);


%double check the analytic and discrete expected values line up
exp_q=exp_q'
var=[ 33/50, 417/850, 62021/104250, 2051429/3101050, 508446627/718000150, 18879487363/25422331350, 3630445108463/4719871840750, 5737845310119/7260890216926, 14040950918611941/20941867468596250, 20280190875861709/28081901837223882, 8641218052570590138563/11534358560646346993750, 332153026287936030235299/432060902628529506928150, 13029549733668826544181379/16607651314396801511764950, 519783033143561147652043363/651477486683441327209068950, 388209564984535037522424009311/559701427544490763117358744750, 4797191009981549300605333020607/8574593127997786279746736771950, 2102167428600719130271586967138463/5023944816961595182258066989286190, 60996402480131262847537325485538559/105108371430035956513579348356923150, 288531759472373242966812111649814572653/606539573061189537908077814481538126250, 245316471388690319572456353434575751829/636015627177632589882531405663447107194, 3559067566329552532102177256147458796814219/11232600728932090271414657751580051466743750, 102646147147415556625916824487208110888404587/383676658130126886965624024771629633496476550, 1486741427840346543397622657873975523397152093261/6393444124856682765228338806470589639333637158250, 26945617355967040302150910175210130024250382249837/74337071392017327169881132893698776169857604663050, 719730973002301995598324817195386816139410528606189/2369572701802514343386511135924432307280361120660650, 22212478955868364855960365903492638553539479291191821/82492086440010617389409315936452274557047529602723050, 106735136334864618530396707639968284697067662661592601469/437027154260031330867504887738957361025433364758601410250, 3710510554835324930946764617520201874848073761488918868381/16514600896258335616855409004949453816418285104850440439050, 319058290052721128664359171301920706923110861464982831384984391/1528168182248836325363196707639681219226404723830197599459345150, 27159494554137397738129862595484632595945622762609660146000991/86231970284519223963340316568086677546786719314860224698644430]';
corr(exp_q, var)
