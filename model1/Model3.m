function mdp = Model3(var)

% Actions are: up    (increase the level of X)
%            : keep  (keep the level of X)
%            : down  (decrease the level of X)
Na = [3 1]'; % 3 actions (up) for the level and 1 action (keep) for 
             % the prior of X

% 0utcomes are: dys (dyshomeostasis)
%             : hom (homeostasis)

%--------------------------------------------------------------------------
Nf = [31 31]';

% Factor 1 (f1) includes 15 different states of the
% level of the physiological variable X. (name v_k)
f1 = 1; % coding for readability

% Prior probabilities over Nf(1)=15 states for f1 (priors)
D{f1} = zeros(Nf(f1),1); % high level of X <-> low level of X

% Factor 2 (f2) includes 15 different states of the
% prior of the physiological variable X. (name: p_i)
f2 = 2; % coding for readability

% Prior probabilities over Nf(2)=15 states for f2 (hyper priors!):
D{f2} = ones(Nf(f2),1); % high prior of X <-> low prior of X

%--------------------------------------------------------------------------

% We have 1 outcome modality:
o1 = 1;
% The outcome includes the agent's conclusion about its current situation
% which is graded in 5 steps from high level of homeostasis to low level
% of homeostasis.

% We give low reward for the low level of homeostasis and high reward for
% the high level of homeostasis.
max_level = floor(Nf(1)/2);
C{o1} = (max_level:-1:-max_level)';
             
No = size(C{o1},1);

% Likelihood over outcomes
% p_i to o_1 likelihood conditioned on v_k

for k = 1:Nf(f2)
  for j = 1:Nf(1)
    A{o1}(:,j,k)= zeros(No(1),1);
    A{o1}(abs(k-j)+1,j,k) = 1;
  end
end

% Transition probabilities
%--------------------------------------------------------------------------
% Define shifts(factor, action)
max_level = floor(Na(1)/2);
shifts = max_level:-1:-max_level;

% Actions for factor 1 (f1)
n = Nf(f1); % number of states for hidden factor f1
for a_i = 1:Na(f1) % for each action
  s = shifts(a_i);
  b = zeros(n,n);
  for i = 1:n
    % Transition probability distribution of action a_i
    temp = normpdf(1:n+2*abs(s),i+abs(s), var);
    % If action is moving up, shift the elements accordingly
    if s>0
      if i<s+1
        boundary = sum(temp(i+abs(s):end));
        b(:,i) =  [boundary, temp(i+abs(s)+1:n+abs(s)+i-1)]';
      else
        b(:,i) =  (temp(2*abs(s)+1:n+2*abs(s)))';
      end
    elseif s==0
      b(:,i) = normpdf(1:n,i, var);
    end
  end
  if s<0
    temp_a = 2*ceil(Na(f1)/2) - a_i;
    b = fliplr(B{f1}(:,:,temp_a));
    b = flip(b,1);
  end
  % Transition matrix of action a_i for f1
  B{f1}(:,:,a_i) = b;
end

% No action for factor 2 (f2)
B{f2}(:,:,Na(f2)) = eye(Nf(2));

% Allowable policies
%--------------------------------------------------------------------------
T = 4; % Time horizon --> T-1 actions per episode
keep = ceil(Na(f1)/2);
for i = 1:T-1
  row = [];
  for j = 1:Na(f1)
    if j ~= keep
      row = cat(2, row, ones(1,T-i)*j, ones(1,i-1)*keep);
    end
  end
  V_p(i,:,f1) = row;
end
V(:,:,f1) = cat(2, V_p(:,:,f1), ones(T-1,1)*keep);
V(:,:,f2) =  1;

% Assign the variables of mdp
mdp.A = A;
mdp.B = B;
mdp.C = C;
mdp.D = D;
mdp.V = V;

return


