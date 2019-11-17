%% Bayesian learner for RL
%real data here:
subject_list= [ 1     2     3     4     5    6     7     8     9    10    11    12    13   14   15    16    17  18    19    20    21       23    24];
%subject_list= [1,2,3,4,5,6,7,  9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,  31,32,33,   35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50  ];
for subject=subject_list;    %loop for all subjects
    subject
    %Import data
    data=readtable(strcat('C:/Users/Peter/Documents/MATLAB/reversal_modelling/sugar_data/', num2str(subject), '.txt'));
    % data=readtable(strcat('E:/Downloads - from SSD/rs_reversal_modelling/', num2str(subject), '.txt'));

    clear C               Q_right                block_history   reward_history
    clear Prob_choice_l   Resp               ntrials         total_Q_left  
    clear Prob_choice_r   response        total_Q_right   
    clear Q_left         block           reward          trial    PDF

    %0. load up reward and responses
    response=data.Var1; % alternatively you can feed it response and reward data here directly
    reward=data.Var2;
    data=double((response==1 & reward==1) | (response==2 & reward==0));
    
    %1. define beta function
    %syms q; %q = probability of 1; conversely 1-q is probability of 0
    q=0.01:0.02:0.99;
    
    %2. for any trial t calculate the updates on Q: q left vs q right separate;
    %q left only updated when the agent learns about LEFT; otherwise left the same;
    %also q right updated only when the agent learns about RIGHT, so L/R are
    %not conjoint. 
    j=[0.001 0.01 0.04 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]; %:0.1:0.99; %Hazard H
    k=[0.5 1 3 5 10 15 20 30 40 50]; %alpha and beta priors
    l=[0.001 0.01 0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5]; %0.005:0.5:5; % beta
    m=-1:0.1:1; %kappa; 0.3 stepsize would be too coarse
    index=allcomb(j,k,k,l,m);
        for l=1:length(index);      %loop for all combinations of the parameters
        H=index(l,1,1);
        memorysize=1; %index(l,2,1);
        alpha=index(l,2,1); %alpha prior!
        beta_prior=index(l,3,1);
        beta=index(l,4,1);
        kappa=index(l,5,1);
        %define memorysize and hazard(H) here
    beta_f=q.^(alpha-1).*(1-q).^(beta_prior-1)*gamma(alpha+beta_prior)/gamma(alpha)/gamma(beta_prior); %fplot(beta, [0 1]);  beta_f=ones(1, length(q))./length(q); %

    %2. define prior on trial 1 as beta function
    prior(1,:)=beta_f;
    
        
           for t=1:length(data) %loop for all trials 
            if t>memorysize;
                data_useful=data(t-memorysize:t);
                k=sum(data_useful); 
            elseif t<=memorysize;
                data_useful=data(1:t);
                k=sum(data_useful); 
            end
            n=length(data_useful); %equivalent to how many trials were completed in total
            %LIKELIHOOD for any trial follows a Binomial distribution
            %Pr(q|Data)=n!/(k!*(n-k)!)*q^(k)*(1-q)^(n-k)
            %likelihood(t+1)=factorial(n)/(factorial(k)*factorial(n-k))*(q^k)*((1-q)^(n-k));
                for i=1:length(q)
                    likelihood(i)=factorial(n)/(factorial(k)*factorial(n-k))*(q(i)^k)*((1-q(i))^(n-k));
                end
                likelihood_t(t+1,:)=likelihood(:);
            %posterior at time t+1 = H*posterior(t)+(1-H)*beta
            posterior(t,:)=likelihood_t(t+1,:) .* prior(t,:);
            nconstant=sum(posterior(t,:));%nconstant=int(posterior(t),0,1);
            posterior(t,:)=posterior(t,:)./nconstant;

            prior(t+1,:)=(1-H)*posterior(t,:) + H*beta_f; % this is the actual posterior of the trial t
            %getting the q estimate from each of the distributions
            

            %EXPECTED VALUE vs HIGHEST PROBABILITY --- q=0:0.01:1; expectedqval(t)=trapz(q,post_function(q))*(1/101)
            %post_func=prior(t+1);post_function=matlabFunction(post_func, 'vars', {q});
            %a=0.01:0.01:0.99;
            %Q as expected value based on the posterior:
            %exp_q(t)=sum(post_function(a)*a')/length(a);
            exp_q(t)=sum(prior(t+1,:)*q');
            %%%%%softmax for probability:
            if response(t)==1;Resp_L=1;Resp_R=0;elseif response(t)==2;Resp_L=0;Resp_R=1;end;
            total_Q_left(t)=exp_q(t);total_Q_right(t)=1-exp_q(t);
Prob_choice_l(t+1)=exp(total_Q_left(t)/beta+Resp_L*kappa)/(exp(total_Q_left(t)/beta+Resp_L*kappa)+exp(total_Q_right(t)/beta+Resp_R*kappa));
Prob_choice_r(t+1)=exp(total_Q_right(t)/beta+Resp_R*kappa)/(exp(total_Q_left(t)/beta+Resp_L*kappa)+exp(total_Q_right(t)/beta+Resp_R*kappa));
            end % end for all trials

                    Prob_sequence=zeros(1, length(response));
                    for trial=2:(length(Prob_choice_l)-1);
                        if response(trial)==1;
                            Prob_sequence(trial)=Prob_choice_l(trial);
                        else
                            Prob_sequence(trial)=Prob_choice_r(trial);        
                        end
                        Prob_sequence(1)=0.5;
                    end
                    %probability of each response in the sequence at each trial
                    PDF(l)=log(prod(Prob_sequence));

                    %random model
                    %r=log(0.5^(length(response)));
                        %Q model fit vs random model
                        %pseudo_R2(l)=(PDF(l)-r)/r;
        end %end for all parameter combinations
%save data for each subject
PDF_all(subject)={PDF};
%pseudo_R2_all(subject)={pseudo_R2};
end %end for all subjects


%% optimizing the parameters

best_fit_params=zeros(5,length(subject_list))';
R2_fit=zeros(1,length(subject_list));
PDF_all_cut=PDF_all;
for subject=subject_list
    for row=1:length(PDF);

        if PDF_all{1,subject}(row) == (max(PDF_all{1,subject}));  %+max(PDF_all{1,subject})*0.0000005);
            best_fit_params(subject,:,:)=(index(row,:,:));
            %R2_fit(subject)=pseudo_R2_all{1,subject}(row);
        end
        
    end
    PDF_all_cut{1,subject}(~isfinite(PDF_all_cut{1,subject}))=[];
    %Model_evidence(subject)=mean(PDF_all_cut{1,subject});
    d(subject)=max(PDF_all{1,subject});
    %BIC needs to be adjusted for a) nr of parameters, here it is 3 and b)
    %nr of responses
    BIC=max(PDF_all{1,subject})-5/2*log(length(response));
    BIC_all(subject)=BIC;
end
%Model_evidence=Model_evidence';
R2_fit=R2_fit';
d=d';
chi2inv(.95, 1)
BIC_all=BIC_all';


%% plotting some of the readouts from the Bayesian learner above

figure; fplot(prior(13:22,:), [0 1])
figure; plot(exp_q); ylim([0 1])
contingency(1:14)=0.85;
contingency(15:30)=0.15;
hold on
plot(contingency, 'r'); hold off;

%% %plotting some data as an example, e.g. total Q values for left
            %and right
            figure(2)
            subplot(2,1,1)
            plot(total_Q_left)
            hold on
            plot(total_Q_right)
            xlabel('Trial Number')
            ylabel('Q value')
            
        %subject 21
                reward_all=zeros(1, length(response))
                        reward_all(1:20)=2
                        reward_all(21:42)=1
                        reward_all(43:61)=2
                        reward_all(62:73)=1
                        
            %plotting Prob_choice for L and R; together with observed
            %responses - see how good your params are at predicting data
            %bear in mind this is for the last set of params
            figure(2); subplot(2,1,2)
            plot(Prob_choice_l)
            hold on
            plot(Prob_choice_r)
            hold on
            plot(response)
            hold on
            xlabel('Trial Number')
            ylabel('P(Response)       Response')
            hold on
            plot(reward_all)
                 
                    
             
             
            
