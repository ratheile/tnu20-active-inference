function mdp = dyshomeostasisModel
    
  N_p = 32; p_min = -2; p_max = 2;
  N_v = 32; v_min = -2; v_max = 2;
  N_a = 5;  a_min = 0.5; a_max = 3;

  p = linspace(p_min, p_max, N_p);
  v = linspace(v_min, v_max, N_p);
  a = linspace(a_min, a_max, N_a); % capacitor tau

  levels = cell(1, N_a);
  for i = 1:N_a
    levels{i} = {append('+', num2str(a(i)))};
  end

  % Factor 1: setpoint + current value
  f1 = 1; % coding for readability
  N_f1 = N_p * N_v; % cartinality c(.)  c(p) * c(v)
  D{f1} = ones(N_f1,1) / N_f1; 

  % Factor 2: allostatic control level
  % f2 = 2; % coding for readability
  % N_f2 = N_a;
  % D{f2} = ones(N_f1,1); 

  % Factor 3: metacognition: certaincy about allostatic level
  f3 = 3; % maybe?

  label.factor{f1} = 'prior+value'; 
  % label.factor{f2} = 'allostatic_control';

  % Outcome Modality 1: Dyshomeostasis
  o1 = 1;
  
  o1_levels = zeros(N_p,N_v);
  for i = 1:N_p
    for j = 1:N_v
      o1_levels(i,j) = v(j) - p(i);
    end
  end
  
  label.modality{o1} = 'state'; 
  o1_levels = abs(o1_levels);
  o1_bounds = [min(o1_levels, [], 'all')  max(o1_levels, [], 'all')];
  o1_levels = (o1_bounds(1):o1_bounds(2));
  C{o1} = o1_levels;
  N_o1 = length(C{o1});
  
  % The outcome includes the agent's conclusion about its current situation which is either
  % being in balance (homeostasis) or being imbalanced (dyshomeostasis).
  % We give low reward for dyshomeostasis log(-4) and high reward for homeostasis log(4)
  
  % Outcome Modality 2: Stress
  % o2 = 2;
  %N_o2 = 3
  % C{o2} = [4 0 -4]
  
  
  %  Outcome: Dishomeostasis
  %--------------------------------------------------------------------------
  sigma = 0.4;
  
  %label.outcome = {};
  for k = 1:N_o1
   % label.outcome(k) = {append('+', num2str(o1_levels(k)))}
    for i = 1:N_p
        for j = 1:N_v
          dv = v(j) - p(i);
          y = normpdf(dv, o1_levels(k), sigma);
          A{o1}(k,(j - 1)*N_p + i) = y;
      end
    end
  end


  % Transition probabilities for Factor 2:
  %--------------------------------------------------------------------------
  shifts = [1 0 -1];
  % define shifts(factor, action)
 % n = N_f1; % number of states for hidden factor f_i
 % for a_i = 1:Na(f_i) % for each action
 %   s = shifts(a_i);
 %
 %   % we shift up or down depending on the transition probs
 %   b = circshift(eye(n), s, 2); % 2 = second dimension (up down)
 %
 %   % the shift is bounded by the highest / lowest state
 %   % there we just remain in that state
 %   overflow_bound = zeros(n, abs(s));
 %   if s > 0 % shift up
 %     overflow_bound(1,:) = 1;
 %     b(:,1:s) = overflow_bound;
 %   elseif s < 0 % shift down
 %     overflow_bound(n,:) = 1;
 %     b(:,n+s+1:n) = overflow_bound;
 %   end
 %   B{f2}(:,:,a_i) = b;
 % end
 
  % smoothing matrix to simulate noise
  %--------------------------------------------------------------------------
  K     = toeplitz(sparse(1,[1 2],[1 1/2],1,N_v));
  K     = K + K';
  K     = K*diag(1./sum(K,1));

  % Transition probabilities for Factor 1:
  %--------------------------------------------------------------------------
  %B{f1}  = sparse(N_f1, N_f1, N_a);
  B{f1}  = zeros(N_f1, N_f1, N_a);
  for k = 1:N_a
      for i = 1:N_p
          for j = 1:N_v
              
              % simulate dynamical System for each state
              %--------------------------------------------------------------
              % compute change in state space
              ds = Funcs.adaptive_system(p(i), v(j), a(k));
              
              % transition probabilities - adaptive value
              %--------------------------------------------------------------
              dv = v - (v(j) + ds(1));
              ii = find(dv > 0,1);
              if ii == 1, % lower bound
                  pv = sparse(1,1,1,N_v,1);
              elseif isempty(ii) % upper bound
                  pv = sparse(N_v,1,1,N_v,1);
              else % linear interpolation
                  ii = [ii - 1,ii];
                  pv = pinv([v(ii); 1 1])*[(v(j) + ds(1)); 1];
                  pv = sparse(ii,1,pv,N_v,1);
              end
              
              % transition probabilities - prior
              %--------------------------------------------------------------
              dp = p - (p(j) + ds(2));
              pp = sparse(i, 1, 1, N_p, 1); % deterministic prior

              
              % place in P
              %--------------------------------------------------------------
              pv     = K*pv;
              p_pv      = pp*pv';
              B{f1}(:,(j - 1)*N_p + i, k) = p_pv(:);
              
          end
      end
  end


  % graphics (transition probabilities)
  %--------------------------------------------------------------------------
  spm_figure('GetWin','Figure B');clf
  
  for i = 1:N_a
    subplot(N_a,1,i);
    imagesc(B{f1}(:,:,i));
    xlabel('pv','FontSize',12);
    ylabel('pv','FontSize',12);
  end
  drawnow;

  mdp.A = A;
  mdp.B = B;
  mdp.C = C;
  mdp.D = D;
  mdp.label = label;
  mdp.dys_actions = a;
return


