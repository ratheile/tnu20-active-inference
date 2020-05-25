function mdp = dyshomeostasisModel


% Actions are: varU  (increase the level of X)
%            : varD  (decrease the level of X)
%            : prU   (increase the prior of X)
%            : prD   (decrease the prior of X)
%            : keep  (keep the prior or level of X)
Na = [3 3]';

% 0utcomes are: dys (dyshomeostasis)
%             : hom (homeostasis)


% Factor 1 (f1) includes 5 different interoceptive states 
% about the prior of the physiological variable. (name: p_i)
f1 = 1; % coding for readability

% Prior probabilities over Nf(1)=5 factors (hyper priors!):
D{f1} = [1 1 1 1 1]'; % high quantity of X (qoX) state <-> low qoX state


% Factor 2 (f1) includes 5 different interoceptive states 
% of the level of the physiological variable. (name v_k)
f2 = 2; % coding for readability


levels = {'+2', '+1', '0', '-1', '-2' };
label.factor{f1} = 'prior'; label.name{f1} = levels;
label.factor{f2} = 'physiological'; label.name{f2} = levels;

% Prior probabilities over Nf(2)=5 factors (priors)
D{f2} = [1 1 1 1 1]'; % high quantity of X (qoX) state <-> low qoX state

Nf = [size(D{f1},1) size(D{f2},1)]';

% NOT USED RIGHT NOW:
% Precision of 3 different interoceptive states
p_I = 0.99; % intH, intM, intL
% Precision of 3 different exteroceptive states
p_E = 0.95; % envH, envM, envL

% We have 1 outcome modality:
o1 = 1;
% The outcome includes the agent's conclusion about its current situation which is either
% being in balance (homeostasis) or being imbalanced (dyshomeostasis).

% We give low reward for dyshomeostasis log(-4) and high reward for homeostasis log(4)
C{o1} = [4 -4]';

label.modality{o1} = 'state'; label.outcome = {'hom', 'dys'};

% Likelihood over outcomes
%--------------------------------------------------------------------------
% p_i to o_1 likelihood conditioned on v_k


% Deterministic A
% for k = 1:Nf(f2)
%   A{o1}(:,:,k)= [zeros(1,Nf(f2)); ones(1,Nf(f2))];
%   A{o1}(1,k,k) = 1; % example: [0 1 0 0 0 ] k = 2
%   A{o1}(2,k,k) = 0; % example: [1 0 1 1 1 ]
% end

for k = 1:Nf(f2)
  sigma = 0.4;
  x = 1:1:Nf(f2);
  y = normpdf(x,k,sigma);
  A{o1}(:,:,k)= [y; 1-y];
end


% Transition probabilities
%--------------------------------------------------------------------------
% define shifts(factor, action)
shifts = [1 0 -1];

for f_i = 1:size(Nf,1)  % for each factor
  n = Nf(f_i); % number of states for hidden factor f_i
  for a_i = 1:Na(f_i) % for each action
    s = shifts(a_i);

    % we shift up or down depending on the transition probs
    b = circshift(eye(n), s, 2); % 2 = second dimension (up down)

    % the shift is bounded by the highest / lowest state
    % there we just remain in that state
    overflow_bound = zeros(n, abs(s));
    if s > 0 % shift up
      overflow_bound(1,:) = 1;
      b(:,1:s) = overflow_bound;
    elseif s < 0 % shift down
      overflow_bound(n,:) = 1;
      b(:,n+s+1:n) = overflow_bound;
    end
    B{f_i}(:,:,a_i) = b;
  end
end


% Allowable policies
%--------------------------------------------------------------------------

V(:,:,f1) =  [2 2 2 2 ;
              1 2 2 2 ;
              1 2 2 2 ;
              1 2 2 2 ;
              1 2 2 2 ;
              3 2 2 2 ;
              3 2 2 2 ;
              3 2 2 2 ;
              3 2 2 2 ;
              ]';
         
V(:,:,f2) =  [2 1 1 1 1 3 3 3 3;  
             2 2 1 1 1 2 3 3 3;
             2 2 2 1 1 2 2 3 3;
             2 2 2 2 1 2 2 2 3];
                

mdp.A = A;
mdp.B = B;
mdp.C = C;
mdp.D = D;
mdp.V = V;
mdp.T = 40; % Max dist. to high reward is 4
mdp.label = label;



return


