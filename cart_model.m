function MDP = cart_model

rng default

% Model Hyperparameter
% --------------------
ops = 3; % Outcome Preference Strength
f1 = 1; % factor 1 alias
f2 = 2; % factor 2 alias
o1 = 1; % outcome 1 alias

Nf1 = 2;
Nf2 = 3;

% Prior beliefs about initial states
% ----------------------------------

% Factors:
% Cart Direction (west, east)
% Pole Orientation (left, up, right)


D{f1} = [1 1] % We dont care about initial cart direction
D{f2} = [0.1 1 0.1] % We assume an upright pole position

% Outcome states:
% Pole  Cart
% left  west
% left  east
% up    west
% up    east
% right west
% right east

V(:,:,f1) = [ 1 1 1 1 1; % strong west (-10 N)
      2 1 1 1 1;
      2 2 1 1 1;
      2 2 2 1 1;
      2 2 2 2 1; 
      2 2 2 2 2]'; % strong east (+10 N)

V(:,:,f2) = 1; % we cannot change the pole position actively
  
% f2 = left
A{o1}(:,:,1) = [
% west east (f1) 
                % Outcome states:
                % Pole  Cart
 1 0            % left  west
 0 1            % left  east
 0 0            % up    west
 0 0            % up    east
 0 0            % right west
 0 0            % right east
];

% f2 = up
A{o1}(:,:,2) = [
% west east (f1) 
                % Outcome states:
                % Pole  Cart
 0 0            % left  west
 0 0            % left  east
 1 0            % up    west
 0 1            % up    east
 0 0            % right west
 0 0            % right east
];

% f2 = right
A{o1}(:,:,3) = [
% west east (f1) 
                % Outcome states:
                % Pole  Cart
 0 0            % left  west
 0 0            % left  east
 0 0            % up    west
 0 0            % up    east
 1 0            % right west
 0 1            % right east
];


S = [0.3 0.5 0.1 0.3]; % Output of Continuous Simulation x dx a da
% Factor 1 Actions
% --------------------------------------------

% control state: west
B{f1}(:,:,1) = [
    % west east 
               % west
               % east
];


% control state: west
B{f1}(:,:,1) = [
    % west east 
               % west
               % east
];

% Factor 2 Actions
% --------------------------------------------
B{f2} = eye(Nf2);

% Prior Preferences
C{o1} = [0 0 ops ops 0 0];


mdp.A = A;
mdp.B = B;
mdp.C = C;
mdp.D = D;
mdp.V = V;

mdp.label.modality = {'State'};

return