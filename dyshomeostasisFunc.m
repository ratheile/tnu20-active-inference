classdef dyshomeostasisFunc

  methods (Static=true)

    function state = adaptive_system(prior, value, action)
      dt = 1;
      dp = prior - value;
      dv = dp * exp(-dt / action);
      state = [dv; prior];
    end

    
    function s = encode_pv(prior, value, Nf)
        s = (value-1)* Nf + prior;
    end

    
    function s = decode_pv(state, Nf)
        prior = mod(state, Nf);
        value = state - prior /  Nf;
        s = [prior value];
    end
    
  end
end