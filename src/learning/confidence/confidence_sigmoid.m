function probability = confidence_sigmoid(logit)
%CONFIDENCE_SIGMOID Numerically stable logistic transform.

probability = zeros(size(logit), 'like', logit);
positive = logit >= 0;
probability(positive) = 1 ./ (1 + exp(-logit(positive)));
negative = ~positive;
exponential = exp(logit(negative));
probability(negative) = exponential ./ (1 + exponential);
end
