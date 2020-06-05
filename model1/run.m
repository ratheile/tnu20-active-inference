rng(333,'twister');

%Definition of Variances
highVar = 1;
lowVar  = 0.5;
noVar   = 0;

%% Run for Model1: Agent in Perfect Condition
% No variance
[f1, f2, f2_post] = Trial("Model1", noVar);
fig = figure(1);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "PerfectAgent.png")

%% Run for Model2: Agent with Abnormal Sensory Signals
% High variance
[f1, f2, f2_post] = Trial("Model2", highVar);
fig = figure(2);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "AgentWithHighSensorVariance.png")

% Low variance
[f1, f2, f2_post] = Trial("Model2", lowVar);
fig = figure(3);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "AgentWithLowSensorVariance.png")

%% Run for Model3: Agent with Abnormal Motor Reflexes
% High variance
[f1, f2, f2_post] = Trial("Model3", highVar);
fig = figure(4);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "AgentWithHighMotorVariance.png")

% Low variance
[f1, f2, f2_post] = Trial("Model3", lowVar);
fig = figure(5);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "AgentWithLowMotorVariance.png")

%% Run for Model4: Agent with Abnormal Reward Sensitivity
[f1, f2, f2_post] = Trial("Model4", highVar);
fig = figure(6);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "AgentWithHighRewardVariance.png")

% Low variance
[f1, f2, f2_post] = Trial("Model4", lowVar);
fig = figure(7);
subplot(3,1,1);
plot(1:size(f2,1), f2,'LineWidth',1.5);
title('Context')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,2);
plot(1:size(f2_post,1), f2_post,'LineWidth',1.5);
title('Homeostatic Setpoint')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
subplot(3,1,3);
plot(1:size(f1,1), f1,'LineWidth',1.5);
title('Bodily State (x)')
ylabel('Level')
xlabel('Time (t)')
xlim([0 610])
yticks(-20:10:20)
saveas(fig, "AgentWithLowRewardVariance.png")