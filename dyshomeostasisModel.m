function mdp = dyshomeostasisModel
    
  N_p = 32; p_min = -2; p_max = 2;
  N_v = 32; v_min = -2; v_max = 2;
  N_a = 5;  a_min = 0.05; a_max = 0.15;

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

  % Factor 2: metacognitive decision making
  % 1) quickly adapt
  % 2) helplessness: "change does not matter"
  f2 = 2; % coding for readability
  N_f2 = 2;
  D{f2} = ones(N_f2,1); 


  label.factor{f1} = 'prior+value'; 
  label.factor{f2} = 'metacognition';

  % Outcome Modality 1: Dyshomeostasis
  o1 = 1;
  o1_utility_disabled = -4; %  TODO: 4 for uniform
  o1_utility_disabled_offset = -2;
  o1_utility_min = 4;
  o1_utility_max = -2;
  o1_utility_levels = 5;
  label.modality{o1} = 'state'; 
  
  o1_levels = zeros(N_p,N_v);
  for i = 1:N_p
    for j = 1:N_v
      o1_levels(i,j) = v(j) - p(i);
    end
  end

  o1_levels = abs(o1_levels);
  o1_bounds = [min(o1_levels, [], 'all')  max(o1_levels, [], 'all')];
  o1_levels = [ ...
    o1_bounds(1) + o1_utility_disabled_offset ...
    linspace(o1_bounds(1), o1_bounds(2), o1_utility_levels) ...
  ];
  C{o1} =  [o1_utility_disabled linspace(o1_utility_min, o1_utility_max, o1_utility_levels)];
  N_o1 = length(C{o1});
  
  % The outcome includes the agent's conclusion about its current situation which is either
  % being in balance (homeostasis) or being imbalanced (dyshomeostasis).
  % We give low reward for dyshomeostasis log(-4) and high reward for homeostasis log(4)
  
  % Outcome Modality 2:  Used in combination with factor 2's "helpless" state
  o2 = 2;
  C{o2} = [-2 4];

  
  % Outcome 1: Perceived Dishomeostasis
  %--------------------------------------------------------------------------
  f2_s = 1; % Punish distance by a lot
  % outcome levels: -1 --- 0 --- 1 ---------------> 4
  % scale:                 0 --- 1 ---------------> 4
  sigma = 0.4;
  %label.outcome = {};
  for k = 1:N_o1
   % label.outcome(k) = {append('+', num2str(o1_levels(k)))}
    for i = 1:N_p
        for j = 1:N_v
          dv = abs(v(j) - p(i));
          y = normpdf(dv, o1_levels(k), sigma);
          A{o1}(k,(j - 1)*N_p + i, f2_s) = y;
      end
    end
  end

  f2_s = 2; % helplessness 
  % outcome levels: -k --- ... --- 0 --- 1 ---------------> 4
  % scale:           0 --> 4
  % In this case, we do not think of eventual dyshomeostasis as something
  % which we should correct. We shift the likelihood towards a neutral state.
  % This is achieved by reducing the perceived distance.
  distance_scale = 1000;
  for k = 1:N_o1
    for i = 1:N_p
        for j = 1:N_v
          dv = abs(v(j) - p(i)) / distance_scale + o1_utility_disabled_offset;
          y = normpdf(dv, o1_levels(k), sigma);
          A{o1}(k,(j - 1)*N_p + i, f2_s) = y;
      end
    end
  end

  % Outcome 2: Disinterest
  %--------------------------------------------------------------------------
  o2_p = 0.1;
  f2_s = 1; % Punish distance by a lot
  for i = 1:N_p
      for j = 1:N_v
        A{o2}(1,(j - 1)*N_p + i, f2_s) = 1-o2_p; % o2 1 -> low reward
        A{o2}(2,(j - 1)*N_p + i, f2_s) = o2_p; % o2 2 -> high reward
    end
  end

  f2_s = 2; % helplessness 
  for i = 1:N_p
      for j = 1:N_v
        A{o2}(1,(j - 1)*N_p + i, f2_s) = o2_p; % o2 1 -> low reward
        A{o2}(2,(j - 1)*N_p + i, f2_s) = 1-o2_p; % o2 2 -> high reward
    end
  end

  % Transition probabilities for Factor 2:
  %--------------------------------------------------------------------------
  p_hl = 0.8;
  p_rec = 0.1;
  B{f2}(:,:,1) = [ 1-p_hl 0; p_hl 1]; % 1 -> 2 % become helpless
  B{f2}(:,:,2) = [ 1 p_rec; 0 1-p_rec]; % 2 -> 1 % become active 
  B{f2}(:,:,3) = [ 1 0; 0 1];  % stay
 
  % smoothing matrix to simulate noise
  %--------------------------------------------------------------------------
  K = toeplitz(sparse(1,[1 2],[1 1/2],1,N_v));
  K = K + K';
  K = K*diag(1./sum(K,1));

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
  mdp.T = 5;
  mdp.label = label;

  mdp.N.f1 = N_f1;
  mdp.N.f2 = N_f2;
  mdp.N.p = N_p;
  mdp.N.v = N_v;
  mdp.N.a = N_a;
  
  % store the discretization spaces
  mdp.d_spaces.a = a;
  mdp.d_spaces.v = v;
  mdp.d_spaces.p = p;
end

