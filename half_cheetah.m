% Half Cheetah Reinforcement Learning Example using DDPG

% Create the Half Cheetah environment
env = rlPredefinedEnv("HalfCheetah-v4");

% Get observation and action info
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Set random seed for reproducibility
rng(0)

% Create actor network
actorNetwork = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(400, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(300, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'ActorFC3')
    tanhLayer('Name', 'ActorTanh')
    scalingLayer('Name', 'ActorScaling', 'Scale', max(actInfo.UpperLimit))];

actorOptions = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'ActorScaling'},actorOptions);

% Create critic network
statePath = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(400, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(300, 'Name', 'CriticStateFC2', 'BiasLearnRateFactor', 0)];
    
actionPath = [
    featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(300, 'Name', 'CriticActionFC1', 'BiasLearnRateFactor', 0)];

commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOptions);

% Create DDPG agent
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',env.Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',128,...
    'DiscountFactor',0.99,...
    'NoiseOptions',rl.option.GaussianActionNoise(...
    'StandardDeviation',0.1,...
    'StandardDeviationDecayRate',1e-5,...
    'StandardDeviationMin',0.01));

agent = rlDDPGAgent(actor,critic,agentOptions);

% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',5000,...
    'MaxStepsPerEpisode',1000,...
    'ScoreAveragingWindowLength',100,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',4000,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',3000);

% Train the agent
trainingStats = train(agent,env,trainOpts);

% Simulate the trained agent
simOpts = rlSimulationOptions('MaxSteps',1000);
experience = sim(env,agent,simOpts);