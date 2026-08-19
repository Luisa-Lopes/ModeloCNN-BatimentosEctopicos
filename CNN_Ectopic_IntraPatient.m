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

%% 2) Divisao intra-paciente
% Para a divisao intra-paciente, cada paciente e separado em conjunto de treino,
% validacao e teste de acordo com a proporcao dos batimentos daquele paciente.

patientList = unique(patientId);
idxTrain = [];
idxVal   = [];
idxTest  = [];

rng(42);

for i = 1:numel(patientList)
    pid = patientList(i);
    idxPaciente = find(patientId == pid);
    labelsPaciente = labels(idxPaciente);

    % Embaralha os indices do paciente
    idxPaciente = idxPaciente(randperm(numel(idxPaciente)));

    % Proporcao 70/15/15 dentro do paciente
    n = numel(idxPaciente);
    nTrain = floor(0.70 * n);
    nVal   = floor(0.15 * n);
    nTest  = n - nTrain - nVal;

    % Ajusta para garantir que exista ao menos 1 amostra em cada conjunto
    if nTrain == 0, nTrain = 1; end
    if nVal == 0 && n > 1, nVal = 1; end
    if nTest == 0 && n > 1, nTest = 1; end

    % Recalcula para garantir que a soma nao exceda o total do paciente
    totalAjustado = nTrain + nVal + nTest;
    while totalAjustado > n
        if nTest > 0
            nTest = nTest - 1;
        elseif nVal > 0
            nVal = nVal - 1;
        else
            nTrain = nTrain - 1;
        end
        totalAjustado = nTrain + nVal + nTest;
    end

    % Se ainda ficou zero em algum conjunto, corrige
    if nTrain == 0 && n > 1, nTrain = 1; end
    if nVal == 0 && n > 2, nVal = 1; end
    if nTest == 0 && n > 3, nTest = 1; end

    % Reajuste final para evitar excesso
    while nTrain + nVal + nTest > n
        if nTest > 0
            nTest = nTest - 1;
        elseif nVal > 0
            nVal = nVal - 1;
        else
            nTrain = nTrain - 1;
        end
    end

    % Garante que o total do paciente continue sendo usado
    if nTrain + nVal + nTest < n
        restante = n - (nTrain + nVal + nTest);
        nTest = nTest + restante;
    end

    % Distribuicao final
    trainIdx = idxPaciente(1:nTrain);
    valIdx   = idxPaciente(nTrain+1:nTrain+nVal);
    testIdx  = idxPaciente(nTrain+nVal+1:end);

    % Caso o ultimo conjunto fique vazio por algum paciente pequeno
    if isempty(testIdx) && numel(idxPaciente) > 1
        testIdx = idxPaciente(end);
        valIdx = setdiff(valIdx, testIdx);
    end

    idxTrain = [idxTrain, trainIdx];
    idxVal   = [idxVal, valIdx];
    idxTest  = [idxTest, testIdx];
end

% Remove possiveis indices duplicados ou vazios
idxTrain = unique(idxTrain(:));
idxVal   = unique(idxVal(:));
idxTest  = unique(idxTest(:));

% Garante separacao correta
if isempty(idxTrain) || isempty(idxVal) || isempty(idxTest)
    error('A divisao intra-paciente gerou algum subconjunto vazio. Verifique a quantidade de amostras por paciente.');
end

%% 3) Monta os tensores de entrada
Xtrain = reshape(beats(:, idxTrain), [181, 1, 1, numel(idxTrain)]);
Xval   = reshape(beats(:, idxVal),   [181, 1, 1, numel(idxVal)]);
Xtest  = reshape(beats(:, idxTest),  [181, 1, 1, numel(idxTest)]);

Ytrain = categorical(labels(idxTrain), [0 1], {'Normal', 'Ectopic'});
Yval   = categorical(labels(idxVal),   [0 1], {'Normal', 'Ectopic'});
Ytest  = categorical(labels(idxTest),  [0 1], {'Normal', 'Ectopic'});
classNames = categories(Ytrain);

%% 4) Arquitetura CNN 1D do artigo ECG Heartbeat Classification Using CNNs
% Quatro convolucoes acopladas (C1-C2 e C3-C4), dois max-poolings
% (S1 e S2), dropout apos C2, C4, F1 e F2, e tres camadas densas
% (a ultima foi ajustada para as duas classes desta base).
layers = [
    imageInputLayer([181 1 1], 'Name', 'input', 'Normalization', 'none')

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
