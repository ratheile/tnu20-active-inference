clear 
clc

rng(5,'twister');

% addpath ~\bayes
% addpath ~\spm\
%addpath ~/ws/mlab/spm12
%addpath ~/ws/mlab/spm12/toolbox/DEM
addpath ./../spm12
addpath ./../spm12/toolbox/DEM

num_trials = 3;
num_episodes = 20;
z  = [4 8 12 16]; % change context for some episodes
pLoc = randi([1 5], 1);
vLoc = randi([1 5], 1);

% With preferences:
trwp = zeros(num_episodes, num_trials);
 for j = 1:num_trials
     
      mdp = {};
      
      for i = 1:num_episodes
             MDP = dyshomeostasisModel();   
            
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             if sum(z==i) == 1
                 pLoc = randi([1 5], 1);
                 vLoc = randi([1 5], 1);
                 [MDP.s]     = [pLoc vLoc]';
             else
                 [MDP.s]     = [pLoc vLoc]';
             end

             MDP  = spm_MDP_VB_X(MDP);     
             spm_figure('GetWin','Figure 1');
             spm_MDP_VB_trial(MDP)
             
             
             mdp{i} = MDP;
             trwp(i,j) = MDP.o(MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end
 
 
 spm_figure("GetWin", "Test");
 