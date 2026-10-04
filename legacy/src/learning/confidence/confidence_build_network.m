function net = confidence_build_network(inputDimension, hiddenWidths, ...
        dropoutRate)
%CONFIDENCE_BUILD_NETWORK Build a binary-logit confidence MLP.

if nargin < 3
    dropoutRate = 0;
end
if dropoutRate < 0 || dropoutRate >= 1
    error('confidence_build_network:BadDropout', ...
        'Dropout rate must be in [0, 1).');
end
dropoutCount = numel(hiddenWidths) * double(dropoutRate > 0);
layerCells = cell(2 * numel(hiddenWidths) + dropoutCount + 2, 1);
layerCells{1} = featureInputLayer(inputDimension, Normalization='none', ...
    Name='confidence_feature');
cursor = 1;
for index = 1:numel(hiddenWidths)
    cursor = cursor + 1;
    layerCells{cursor} = fullyConnectedLayer(hiddenWidths(index), ...
        Name=sprintf('fc%d', index));
    cursor = cursor + 1;
    layerCells{cursor} = swishLayer(Name=sprintf('swish%d', index));
    if dropoutRate > 0
        cursor = cursor + 1;
        layerCells{cursor} = dropoutLayer(dropoutRate, ...
            Name=sprintf('dropout%d', index));
    end
end
cursor = cursor + 1;
layerCells{cursor} = fullyConnectedLayer(1, Name='logit');
layers = vertcat(layerCells{:});
net = dlnetwork(layers);
end
