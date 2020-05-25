clear 
clc

rng(0,'twister');


% addpath ~\bayes
% addpath ~\spm\
addpath ~/ws/mlab/spm12
addpath ~/ws/mlab/spm12/toolbox/DEM

num_trials = 1;
num_episodes = 10;
z  = [1 4 7]; % change context for some episodes
r = randi([1 5],1,100);
loc1 = 5;
loc2 = 1;

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
                 loc1 = r(i+1);
                 loc2 = r(i);
                 [MDP.s]     = [3 loc1]';
             else
                 [MDP.s]     = [3 loc2]';
             end

             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i} = MDP;
             trwp(i,j) = MDP.o(MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end
 