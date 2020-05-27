clear 
clc

rng(5,'twister');

addpath ./../spm12
addpath ./../spm12/toolbox/DEM

num_trials = 1;
num_episodes = 20;
change  = [1 4 8 12 16]; % change context for some episodes

contexts = {};
    contexts(1) = {struct('p', 2)};
    contexts(4) = {struct('p', 16)};
    contexts(8) = {struct('p', 5)};
    contexts(12) = {struct('p', 16)};
    contexts(16) = {struct('p', 2)};

% Hyperparameter
Nf = 32;
T = 10;


% With preferences:
trwp = zeros(num_episodes, num_trials);

for j = 1:num_trials
    
    mdp_hist = {};
    X_hist{1} = zeros(Nf * Nf, num_episodes);

    v_start = 2;

    p_true_hist = zeros(num_episodes * T, num_trials);
    p_hist = zeros(num_episodes * T, num_trials);
    v_true_hist = zeros(num_episodes * T, num_trials);
    v_hist = zeros(num_episodes * T, num_trials);
    u_hist = zeros(num_episodes * (T-1), num_trials);
    
    for i = 1:num_episodes

        % Set up the model
        mdp = dyshomeostasisModel();   
        mdp.T = T;


        if ~isempty(find(change == i)) % change context
            p_start = contexts{i}.p;
        end


        if i > 1
            mdp.D{1} = X{1};
        end
        
        % sigma = 1;
        % x = 1:5;
        % y = normpdf(x,v,sigma);
        % MDP.D{2} = y';

        [mdp.s] = [Funcs.encode_pv(p_start, v_start, Nf)];

        MDP  = spm_MDP_VB_X(mdp);
        spm_figure('GetWin','Figure 1');
        spm_MDP_VB_trial(MDP)
        
        mdp_hist{i} = MDP;
        trwp(i,j) = MDP.o(MDP.T);        
        
        % keeping the posterior
        X{1} = MDP.X{1}(:,end);
        X_hist{1}(:,i) = X{1}(:,end);
        % X{2} = MDP.X{2}(:,end);
        
        as_v = v_start;
        as_p = p_start;
        for k = 1:T

            % Copy the "true" state from the agent
            % We want to compare it to the actual truth
            s = Funcs.decode_pv(MDP.s(k), Nf);
            p_hist((i-1)*T+k) = s(1);
            v_hist((i-1)*T+k) = s(2);
            u_hist((i-1)*(T-1)+1:i*(T-1)) = MDP.u;
            
            if k > 1
                kk = k - 1;
                ds = Funcs.adaptive_system(as_p, as_v, mdp.dys_actions(MDP.u(kk)));
                as_v = ds(1) + as_v;
                as_p = ds(2);
                p_true_hist((i-1)*T+k) = as_p;
                v_true_hist((i-1)*T+k) = as_v;
            else
                p_true_hist((i-1)*T+k) = as_p;
                v_true_hist((i-1)*T+k) = as_v;
            end

            if k == 10
                % Carry over the true state to the next episode
                v_start = s(2);
            end

        end

        % Plotting
        r = 1:(i-1)*T;
        spm_figure('GetWin', 'Trajectory')

        subplot(2,2,1);
        plot(r, p_true_hist(r), 'LineWidth', 2); hold on;
        plot(r, v_true_hist(r), 'LineWidth', 2); 

        subplot(2,2,2);
        plot(r, p_hist(r), 'LineWidth', 2); hold on;
        plot(r, v_hist(r), 'LineWidth', 2);

        subplot(2,2,3);
        imagesc(X_hist{1});
        title('Posterior probability');

        subplot(2,2,4);
        plot(r, u_hist(r), 'LineWidth', 2);
        title('Actions');
    end    


end
