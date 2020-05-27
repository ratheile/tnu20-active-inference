classdef Funcs 

  methods (Static=true)

    function state = adaptive_system(prior, value, action)
      dt = 1;
      dp = prior - value;
      dv = dp * exp(-dt / action);
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
  end
end