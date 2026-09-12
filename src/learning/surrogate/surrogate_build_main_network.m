function net = surrogate_build_main_network(cfg)
%SURROGATE_BUILD_MAIN_NETWORK Build the fixed 208-128-128-128-4 MLP.

layers = [
    featureInputLayer(cfg.dataset.featureDimension, ...
        Normalization='none', Name='feature')
    fullyConnectedLayer(cfg.network.width, Name='fc1')
    swishLayer(Name='swish1')
    fullyConnectedLayer(cfg.network.width, Name='fc2')
    swishLayer(Name='swish2')
    fullyConnectedLayer(cfg.network.width, Name='fc3')
    swishLayer(Name='swish3')
    fullyConnectedLayer(cfg.dataset.targetDimension, Name='command')];

if strcmp(cfg.network.outputActivation, 'tanh')
    layers = [layers
        tanhLayer(Name='bounded_output')];
elseif ~strcmp(cfg.network.outputActivation, 'linear')
    error('surrogate_build_main_network:BadOutputActivation', ...
        'Unknown output activation: %s', cfg.network.outputActivation);
end

net = dlnetwork(layers);
learnableCount = sum(cellfun(@numel, net.Learnables.Value));
if learnableCount ~= cfg.network.parameterCount
    error('surrogate_build_main_network:ParameterCountMismatch', ...
        'Expected %d learnable parameters but built %d.', ...
        cfg.network.parameterCount, learnableCount);
end
end
