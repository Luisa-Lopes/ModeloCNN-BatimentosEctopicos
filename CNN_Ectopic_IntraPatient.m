%% CNN para classificacao de batimentos ectopicos (divisao intra-paciente)
% Base: MITBIH_Beats.mat
% Estrutura esperada da base:
%   beats    -> 181 x N
%   labels   -> 1 x N
%   patientId -> 1 x N
%
% Observacao:
%   - A divisao e feita intra-paciente, isto e, batimentos de cada paciente sao
%     separadamente distribuídos entre treino, validacao e teste.

clear; clc;

%% 1) Carrega a base
basePath = 'D:\ProcessamentoDeSinais\TCC\Model\extBati\MITBIH_Beats.mat';
load(basePath);

beats = double(beats);
labels = double(labels(:).');
patientId = double(patientId(:).');

if size(beats, 2) ~= numel(labels)
    beats = beats.';
end

%% 2) Divisao intra-paciente estratificada
% Cada paciente aparece nos tres conjuntos quando possui amostras suficientes.
% A estratificacao e feita separadamente para Normal e Ectopic dentro de cada
% paciente, preservando a proporcao das classes em treino, validacao e teste.

patientList = unique(patientId);
splitFractions = [0.70, 0.15, 0.15];
idxTrain = zeros(numel(labels), 1);
idxVal   = zeros(numel(labels), 1);
idxTest  = zeros(numel(labels), 1);
trainCount = 0;
valCount = 0;
testCount = 0;

rng(42);

for i = 1:numel(patientList)
    pid = patientList(i);
    patientIndices = find(patientId == pid);

    for classValue = [0, 1]
        classIndices = patientIndices(labels(patientIndices) == classValue);
        classIndices = classIndices(randperm(numel(classIndices)));
        nClass = numel(classIndices);

        nTrain = round(splitFractions(1) * nClass);
        nVal   = round(splitFractions(2) * nClass);
        nTest  = nClass - nTrain - nVal;

        % Corrige arredondamentos sem perder ou duplicar amostras.
        while nTest < 0
            [~, largestSplit] = max([nTrain, nVal, nTest]);
            if largestSplit == 1
                nTrain = nTrain - 1;
            else
                nVal = nVal - 1;
            end
            nTest = nClass - nTrain - nVal;
        end

        trainEnd = nTrain;
        valEnd = nTrain + nVal;
        idxTrain(trainCount + (1:nTrain)) = classIndices(1:trainEnd);
        idxVal(valCount + (1:nVal)) = classIndices(trainEnd+1:valEnd);
        idxTest(testCount + (1:nTest)) = classIndices(valEnd+1:end);
        trainCount = trainCount + nTrain;
        valCount = valCount + nVal;
        testCount = testCount + nTest;
    end
end

idxTrain = sort(idxTrain(1:trainCount));
idxVal   = sort(idxVal(1:valCount));
idxTest  = sort(idxTest(1:testCount));

% Garante separacao correta
if isempty(idxTrain) || isempty(idxVal) || isempty(idxTest)
    error('A divisao intra-paciente gerou algum subconjunto vazio. Verifique a quantidade de amostras por paciente.');
end

splitEctopicCounts = [sum(labels(idxTrain) == 1), ...
    sum(labels(idxVal) == 1), sum(labels(idxTest) == 1)];
fprintf('Ectopicos por conjunto: treino=%d, validacao=%d, teste=%d\n', ...
    splitEctopicCounts(1), splitEctopicCounts(2), splitEctopicCounts(3));

if any(splitEctopicCounts == 0)
    error('A divisao intra-paciente deixou algum conjunto sem ectopicos. Ajuste a base ou as proporcoes.');
end

%% 3) Monta os tensores de entrada
Xtrain = reshape(beats(:, idxTrain), [300, 1, 1, numel(idxTrain)]);
Xval   = reshape(beats(:, idxVal),   [ 300, 1, 1, numel(idxVal)]);
Xtest  = reshape(beats(:, idxTest),  [300, 1, 1, numel(idxTest)]);

Ytrain = categorical(labels(idxTrain), [0 1], {'Normal', 'Ectopic'});
Yval   = categorical(labels(idxVal),   [0 1], {'Normal', 'Ectopic'});
Ytest  = categorical(labels(idxTest),  [0 1], {'Normal', 'Ectopic'});
classNames = categories(Ytrain);

%% 4) Arquitetura CNN 1D do artigo ECG Heartbeat Classification Using CNNs
% Quatro convolucoes acopladas (C1-C2 e C3-C4), dois max-poolings
% (S1 e S2), dropout apos C2, C4, F1 e F2, e tres camadas densas
% (a ultima foi ajustada para as duas classes desta base).
layers = [
    imageInputLayer([300 1 1], 'Name', 'input', 'Normalization', 'none')

    convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')

    convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.5, 'Name', 'drop_conv2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')

    convolution2dLayer([3 1], 64, 'Padding', 'same', 'Name', 'conv3')
    reluLayer('Name', 'relu3')

    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'drop_conv4')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')

    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'drop_fc1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu_fc2')
    dropoutLayer(0.5, 'Name', 'drop_fc2')
    fullyConnectedLayer(2, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOut', 'Classes', classNames)
];

%% 5) Opcoes de treinamento
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xval, Yval}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


%% 6) Treinamento
% Fixa a inicializacao da rede para comparacoes reproduziveis.
rng(42);
net = trainNetwork(Xtrain, Ytrain, layers, options);

%% 7) Avaliacao
% Seleciona o limiar da classe Ectopic exclusivamente na validacao,
% maximizando o F1-score. Em caso de empate, usa o valor mais proximo de 0.5.
[~, scoresVal] = classify(net, Xval);
ectopicIdx = find(strcmp(classNames, 'Ectopic'), 1);
thresholds = (0:0.01:1).';
f1Val = zeros(numel(thresholds), 1);
isEctopicVal = Yval(:) == 'Ectopic';

for i = 1:numel(thresholds)
    predEctopicVal = scoresVal(:, ectopicIdx) >= thresholds(i);
    tp = sum(predEctopicVal & isEctopicVal);
    fp = sum(predEctopicVal & ~isEctopicVal);
    fn = sum(~predEctopicVal & isEctopicVal);

    if (2 * tp + fp + fn) > 0
        f1Val(i) = (2 * tp) / (2 * tp + fp + fn);
    end
end

bestF1Val = max(f1Val);
candidateIdx = find(abs(f1Val - bestF1Val) < eps);
[~, nearestToHalf] = min(abs(thresholds(candidateIdx) - 0.5));
bestIdx = candidateIdx(nearestToHalf);
optimalThreshold = thresholds(bestIdx);

% Aplica no teste somente o limiar definido acima.
[~, scoresTest] = classify(net, Xtest);
YPred = categorical(repmat({'Normal'}, numel(Ytest), 1), classNames);
YPred(scoresTest(:, ectopicIdx) >= optimalThreshold) = 'Ectopic';
acc = mean(YPred(:) == Ytest(:));
cm = confusionmat(Ytest, YPred);

fprintf('\nAcuracia no teste: %.4f\n', acc);
fprintf('Limiar otimo para Ectopic na validacao: %.2f (F1 = %.4f)\n', ...
    optimalThreshold, bestF1Val);
disp('Matriz de confusao:');
disp(cm);

%% 8) Exemplo de predicao da primeira amostra de teste
fprintf('Classe verdadeira da 1a amostra de teste: %s\n', string(Ytest(1)));
fprintf('Classe prevista da 1a amostra de teste: %s\n', string(YPred(1)));
