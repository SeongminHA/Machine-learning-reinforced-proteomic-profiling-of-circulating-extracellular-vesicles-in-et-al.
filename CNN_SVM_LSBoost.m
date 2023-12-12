clc
clear all
close all
%% LSBoost_miRNA intensity regression tool learner

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
[trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

template = templateTree(...
    'MinLeafSize', 1, ...
    'NumVariablesToSample', 10);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 19, ...
    'Learners', template, ...
    'LearnRate', 0.2376760912822294);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'ApuVA', 'ApuVA1', 'Concenctrationngml', 'EpV', 'EpV1', 'IpMBuA', 'IpMBuA1', 'IuA', 'VarName4', 'pH'};
trainedModel.RegressionEnsemble = regressionEnsemble;
% Predictor and response variable extraction
% This code processes the data into a form suitable for training a model.
inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

template = templateTree(...
    'MinLeafSize', 6, ...
    'NumVariablesToSample', 8);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 1000, ...
    'Learners', template, ...
    'LearnRate', 0.01);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'ApuVA', 'ApuVA1', 'Concenctrationngml', 'EpV', 'EpV1', 'IpMBuA', 'IpMBuA1', 'IuA', 'VarName4', 'pH'};
trainedModel.RegressionEnsemble = regressionEnsemble;

inputTable = trainingData;
predictorNames = {'pH', 'Concenctrationngml', 'VarName4', 'IpMBuA', 'EpV', 'ApuVA', 'IpMBuA1', 'EpV1', 'ApuVA1', 'IuA'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 50);

% Calculate validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Verification RMSE Calculation
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));

regressionLearner

%% Programmatic Transfer Learning Using Support vector machine
%LoadData
% unzip('MerchData.zip');
imds = imageDatastore('Twoclass', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% Input layer random forest
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,100);

I = imtile(imds, 'Frames', idx);

figure
imshow(I)
%Load Pretrained Network
net = squeezenet;

analyzeNetwork(net)
inputSize = net.Layers(1).InputSize

% Extract Image Features 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool10';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%Fit Image Classifier
mdl = fitcecoc(featuresTrain,YTrain);

%Classify Test Images
YPred = predict(mdl,featuresTest);
% idx = [1 5 10 15 20 25 30 35 40 45 50 60 65 70 74]; 
%  idx = [1 13  25 37 49 61 73 85 97 109 121 133 155 167 179 191];
idx = randi([1,max(size(imdsTest.Files))],1,16);


figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end

accuracy = mean(YPred == YTest)


% rng(0,'twister');
% a = 1;
% b = 73;
% idx = (b-a).*rand(16,1) + a;
% idx = randi([1,677],1,16);
idx= randi([1,max(size(imdsTest.Files))],1,16);

figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end

accuracy = mean(YPred == YTest)

classificationLearner



