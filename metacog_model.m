function mdp = model

clear;
clc;

D{1} = [0 128 0]'; % Discretized Angle

Nf = 1; % Number of Factors
Ns = numel(D{1}); % 3 State Regions


No = [3]; % We have 3 Outcomes for our Angle  


A{1} = ones(No, Ns)

B{1}(:,:,1) = eye(3)
B{1}(:,:,2) = eye(3)


return