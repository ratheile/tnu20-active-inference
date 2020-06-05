function [f1, f2, f2_post] = Trial(method, var)
% Add paths, determine random seed and common parameters
clc;
addpath ./../spm12
addpath ./../spm12/toolbox/DEM
rng(332,'twister');

num_trials = 1;
num_episodes = 200;
% Random change when our prior is at the middle
options = [15 -15]; % Amount of change is either +15 or -15
prior_change = options(randi(numel(options), 1, num_episodes)); % Get some random 1s and 2s
% Episodes where we will change the priors
prior_switch = 20:40:180;
% Random perturbations for the physiological level of X
perturbations=[-13:-10, 10:13];
value_change = perturbations(randi(numel(perturbations), 1, num_episodes));
% Episodes where we will give perturbations to X
value_switch = 5:40:165;
% Store states' path for visualization
f1_path = zeros(num_episodes*3+1, num_trials);
f2_path = zeros(num_episodes*3+1, num_trials);
X_f2_path = zeros(num_episodes*3+1, num_trials);

 for j = 1:num_trials
   % Initial positions
   f1_loc = 16;
   f2_loc = 16;
   % Learn over the different episodes (i.e. steps)
   for i = 1:num_episodes
     % Different experiments with different abnormalities
     if method == "Model1"
       MDP = Model1(); % No argument (no disease)
     elseif method == "Model2"
       MDP = Model2(var); % Give a variance value (sensory disease)
     elseif method == "Model3"
       MDP = Model3(var); % Give a variance value (motor disease)
     elseif method == "Model4"
       MDP = Model4(var); % Give a variance value (abnormal reward sensitivity)
     end
   
     % Use the posterior from previous trial
     if i > 1
       MDP.D{2} = f2_posterior;
     end
     % Prior change in some episodes
     if sum(prior_switch==i) == 1
       if f2_loc>16
         f2_loc = f2_loc - 15;
       elseif f2_loc<16
         f2_loc = f2_loc + 15;
       else
         f2_loc = f2_loc + prior_change(i);
       end
     end
     % Perturb the value of X in some episodes
     if sum(value_switch==i) == 1
       temp = value_change(i);
       if f2_loc == 31 && sign(temp) == 1
         temp = -temp;
       end
       if f2_loc == 1 && sign(temp) == -1
         temp = -temp;
       end
       f1_loc = f1_loc + temp;
       f1_loc = max(1, f1_loc);
       f1_loc = min(f1_loc, 31);
     end

     % True states for the initialization
     MDP.s = [f1_loc f2_loc]';
     
     % Initial state probability of f1
     MDP.D{1}(f1_loc,1) = 1;

     % Solve active inference problem
     MDP = spm_MDP_VB_X(MDP); 
     
     % Keep the posterior of f2
     f2_posterior = MDP.X{2}(:,end);
     
     % Keep the last level of f1
     f1_loc = MDP.s(1,end);
     
     % Save states path for visualization
     if i == 1
       f1_path(i:i+3,j) = MDP.s(1,:);
       f2_path(i:i+3,j) = MDP.s(2,:);
       [~, idx] = max(MDP.X{2},[],1);
       X_f2_path(i:i+3,j) = idx;
     else
       f1_path(i*3-1:i*3+1,j) = MDP.s(1,2:end);
       f2_path(i*3-1:i*3+1,j) = MDP.s(2,2:end);
       [~, idx] = max(MDP.X{2}(:,2:end),[],1);
       X_f2_path(i*3-1:i*3+1,j) = idx;
     end
   end
 end
f1 = f1_path(:,1) - 16;
f2 = f2_path(:,1) - 16;
f2_post = X_f2_path(:,1) -16;

return
