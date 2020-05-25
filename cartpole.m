env = rlPredefinedEnv("CartPole-Discrete");
schedule = [1 2 1 2 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2];


s = sensorModel(env);

for i = 1:length(schedule)
    
    actions = [-10 10];
    [observation, reward, isDone] = step(env, actions(schedule(i)));
    state = env.State;
    x = state(1);
    dx = state(2);
    a = state(3);
    da = state(4);
    plot(env);
end




function p = policies()
    p = unique(nchoosek(repmat('LR', 1,4), 4), 'rows');
end


function sensor_state = sensorModel(env)
    tshd = 10;
    a = env.State(3);
    if (a <= -tshd) 
        sensor_state = 1;
    elseif (a <= tshd)
        sensor_state = 2;
    else % (a >= tshd)
        sensor_state = 3;
    end

end