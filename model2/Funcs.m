classdef Funcs 

  methods (Static=true)

    function state = adaptive_system(prior, value, action)
      dp = prior - value;
      dv = dp * action;
    noise = (normrnd(0, action / 2));
      dv = dv + noise;
      state = [dv; prior];
    end

    function s = encode_pv(prior, value, N)
        s = (value - 1) * N + prior;
    end

    function s = decode_pv(state, N)
        prior = mod(state, N);
        value = (state - prior) / N;
        s = [prior value + 1];
    end

    function s = decode_cont(value, d_space)
      s = find((d_space - value) >= 0,1);
      if isempty(s)
          s = length(d_space);
      end
    end

  end
end