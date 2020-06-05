classdef Funcs 

  methods (Static=true)

    % a simple distretized first order ODE to simulate a continuous signal
    function state = adaptive_system(prior, value, action)
      dp = prior - value;
      dv = dp * action;
    noise = (normrnd(0, action / 2));
      dv = dv + noise;
      state = [dv; prior];
    end

    % map from (p,v) -> f1 state
    function s = encode_pv(prior, value, N)
        s = (value - 1) * N + prior;
    end

     % map from f1 -> (p_tilde,v_tilde) state
    function s = decode_pv(state, N)
        prior = mod(state, N);
        value = (state - prior) / N;
        s = [prior value + 1];
    end

     % map from f1 -> (p,v) state (continuous)
    function s = decode_cont(value, d_space)
      s = find((d_space - value) >= 0,1);
      if isempty(s)
          s = length(d_space);
      end
    end

  end
end