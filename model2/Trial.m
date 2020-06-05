clear 
clc

rng(8,'twister');

addpath ./../spm12
addpath ./../spm12/toolbox/DEM

num_trials = 1;
num_episodes = 30;

% change episodal context for some episodes
change  = [1 2 3 4 6 12 14 15 16 17 18]; 

% These are the perturbations applied during the trial
contexts = {};
    contexts(1) = {struct('v', 2)}; % set the value to 2
    contexts(2) = {struct('v', 28)}; % set the value to 28
    contexts(3) = {struct('v', 28)};
    contexts(4) = {struct('v', 28)};
    contexts(6) = {struct('v', 5)};
    
    contexts(12) = {struct('v', 28)};

    contexts(14) = {struct('v', 28)};
    contexts(15) = {struct('v', 28)};
    contexts(16) = {struct('v', 28)};
    contexts(17) = {struct('v', 28)};
    contexts(18) = {struct('v', 28)};

% Hyperparameter (carry over some predefined ranges of the model)
cfg_model = Model(); % used for setup, never evaluated
N_F = 2; f1 = 1; f2 = 2; % we employ 2 factors 
T = cfg_model.T;
N_f1 = cfg_model.N.f1;
N_f2 = cfg_model.N.f2;
N_v = cfg_model.N.v;
N_a = cfg_model.N.a;

% With preferences:
trwp = zeros(num_episodes, num_trials);

for j = 1:num_trials
    
    % MDP variables, in discrete space
    mdp_hist = {};
    X_hist{f1} = zeros(N_f1, num_episodes * T);
    X_hist{f2} = zeros(N_f2, num_episodes * T);
    P_hist{f1} = zeros(N_a, num_episodes * T);
    P_hist{f2} = zeros(3, num_episodes * T);
    
    p_start = 2;
    v_start = 2;
    f2_start = 1;

    p_hist = zeros(num_episodes * T, num_trials);
    v_hist = zeros(num_episodes * T, num_trials);
    u_hist = zeros(num_episodes * (T-1), N_F, num_trials);
    
    % Adaptive System variables. After an episode we simulate
    % The worlds true state (ex. an ODE system)
    as_v = zeros(num_episodes * T, num_trials);
    as_p = zeros(num_episodes * T, num_trials);
    p_true_hist = zeros(num_episodes * T, num_trials);
    v_true_hist = zeros(num_episodes * T, num_trials);

    % need to know the exact discretization levels from the model
    d_space_v = cfg_model.d_spaces.v;
    d_space_p = cfg_model.d_spaces.p;
    d_space_a = cfg_model.d_spaces.a;
    
    for i = 1:num_episodes
        
        % Set up the model
        mdp = Model();   
        
        % Take over values from prev episode
        % -----------------------------------------------------------------
        if i > 1
            mdp.D{f1} = X{f1}; % Copy belief about our states
            mdp.D{f2} = X{f2}; % Copy metacognitive belief 
            
            % set continuous variables
            % those depend on actions (which we have T-1)
            % therefore we set the initial value equal to the last one
            % to have matchimagescing array sizes
            as_v((i-1)*T+1) = as_v((i-2)*T+T);
            as_p((i-1)*T+1) = as_p((i-2)*T+T);
        else
            % initialize continuous system at episode 1
            as_v(1) = d_space_v(v_start);
            as_p(1) = d_space_p(p_start);
        end
        
        % Experimental input (perturbations)
        % -----------------------------------------------------------------
        if ~isempty(find(change == i)) % change context
            v_start = contexts{i}.v; 
            as_v((i-1)*T+1) = d_space_v(v_start);
        end

        f1_start = Funcs.encode_pv(p_start, v_start, N_v);
        [mdp.s] = [ f1_start; f2_start ];

        MDP  = spm_MDP_VB_X(mdp);
        spm_figure('GetWin','Figure 1');
        spm_MDP_VB_trial(MDP)
        
        mdp_hist{i} = MDP;
        trwp(i,j) = MDP.o(MDP.T);        
        % keeping the posterior
        X{f1} = MDP.X{f1}(:,end);
        X{f2} = MDP.X{f2}(:,end);
        
        % store for analysis + plots
        X_hist{f1}(:, (i-1)*T+1:(i-1)*T+T) = MDP.X{f1};
        X_hist{f2}(:, (i-1)*T+1:(i-1)*T+T) = MDP.X{f2};
        
        u_hist((i-1)*(T-1)+1:i*(T-1),: ,j ) = MDP.u'; 
        
        
        % States of which the agent thinks they are the "truth"
        % We want to compare it to the actual truth. Therefore
        % we simulate all the actions in "replay mode" using
        % the true underlying system.
        s_f1 = MDP.s(f1, :);
        s_f2 = MDP.s(f2, :);
        
        
        for k = 1:T
            i_t = (i-1)*T+k;
            
            % Collect some stats
            % --------------------------------------------------------------
            % Decode the the compined p-v state space
            s1 = Funcs.decode_pv(s_f1(k), N_v);
            p_hist(i_t) = s1(1);
            v_hist(i_t) = s1(2);
            
            %  Handle the continuous simulation for T-1 steps
            % --------------------------------------------------------------
            if k > 1
                kk = k - 1;
                % take the action in the real world that the agent choose to be best
                ds = Funcs.adaptive_system( ...
                as_p(i_t-1), ...
                as_v(i_t-1), ...
                d_space_a(MDP.u(f1, kk)) ...
                );
                as_v(i_t) = ds(1) + as_v(i_t-1);
                as_p(i_t) = ds(2);

                P_hist{f1}(:,i_t) = sum(MDP.P(:,:,kk), 2);
                P_hist{f2}(:,i_t) = sum(MDP.P(:,:,kk), 1);
            end

            if k == T
                % Carry over the true state to the next episode
                v_start = Funcs.decode_cont(as_v(i_t), d_space_v);
                p_start = Funcs.decode_cont(as_p(i_t), d_space_p);
                f2_start = s_f2(k);
            end

        end

        % Plotting
        r = 1:(i-1)*T;
        plt_x = 4; plt_y = 1;
        spm_figure('GetWin', 'Trajectory');

        subplot(plt_x,plt_y,1);
        plot(r, as_p(r), 'LineWidth', 2); hold on;
        plot(r, as_v(r), 'LineWidth', 2); 
        title('Continuous Dynamics Trajectory (ODE)', 'Interpreter', 'tex');
        ylabel('State (v,p)', 'Interpreter', 'tex');
        xlabel('Timestep')

        subplot(plt_x,plt_y,2);
        plot(r, p_hist(r), 'LineWidth', 2); hold on;
        plot(r, v_hist(r), 'LineWidth', 2);
        title('Agent�s Perceived Trajectory (decoded f_1)' , 'Interpreter', 'tex');
        ylabel('Decoded State (v,p)', 'Interpreter', 'tex');
        xlabel('Timestep')

        subplot(plt_x,plt_y,3);
        imagesc(P_hist{f1});
        title('Actions f_1', 'Interpreter', 'tex');
        ylabel('Action a_{f_1}', 'Interpreter', 'tex');
        xlabel('Timestep', 'Interpreter', 'tex');

        subplot(plt_x,plt_y,4);
        imagesc(P_hist{f2});
        title('Actions f_2', 'Interpreter', 'tex');
        ylabel('Action a_{f_2}', 'Interpreter', 'tex');
        xlabel('Timestep', 'Interpreter', 'tex');
        
        
        plt_x = 3; plt_y = 1;
        spm_figure('GetWin', 'Posteriors');
        xlabel('Timestep', 'Interpreter', 'tex');

        subplot(plt_x,plt_y,1);
        imagesc(log(X_hist{f1}+0.00001));
        title('Log Posterior probability of f_1', 'Interpreter', 'tex');
        ylabel('log(q(s))', 'Interpreter', 'tex');
        xlabel('Timestep', 'Interpreter', 'tex');
        
        subplot(plt_x,plt_y,2);
        imagesc(log(X_hist{f1}(400:600,:)+0.00001));
        title('Log Posterior probability of f_1 (400-600)', 'Interpreter', 'tex');
        ylabel('log(q(s))', 'Interpreter', 'tex');
        xlabel('Timestep', 'Interpreter', 'tex');

        subplot(plt_x,plt_y,3);
        imagesc(X_hist{f2});
        title('Posterior probability', 'Interpreter', 'tex');
        ylabel('q(s)', 'Interpreter', 'tex');
        xlabel('Timestep', 'Interpreter', 'tex');
    end    
end
