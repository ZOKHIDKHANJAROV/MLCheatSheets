# 📚 Что изучить заранее

Зависимости между темами: для каждой темы — шпаргалки, которые стоит изучить **до** неё.
Учебный портал показывает их при просмотре темы («Сначала: …») вместе с обратными связями («Дальше: …»).

## Формат

Одна строка — одна тема: `- тема: пред1, пред2`, где имена — это имена шпаргалок без `_cheatsheet.html`
(для `cheatsheets/linreg_cheatsheet.html` — `linreg`). Указывайте только **прямые** зависимости —
транзитивные портал выведет сам. Циклы запрещены: `python tools/check_project.py` их найдёт.

## Основы и подготовка данных

- pandas_ml: numpy_ml
- matplotlib_seaborn_ml: numpy_ml, pandas_ml
- bias_variance: overfitting_underfitting
- cross_validation: overfitting_underfitting
- gradient_descent: numpy_ml
- preprocessing: pandas_ml
- data_preprocessing_missing_encoding: preprocessing
- outliers_handling: preprocessing
- categorical_encoding: preprocessing
- scaling_normalization: preprocessing
- dimensionality_reduction: scaling_normalization

## Классические алгоритмы

- linreg: gradient_descent
- regression_metrics: linreg
- polynomial_regression: linreg, overfitting_underfitting
- ridge_lasso_elasticnet: linreg, bias_variance, scaling_normalization
- logreg: linreg
- confusion_matrix: logreg
- classification_metrics: confusion_matrix
- roc_auc: classification_metrics
- knn: scaling_normalization, classification_metrics
- naive_bayes: classification_metrics
- svm: logreg, scaling_normalization
- kmeans: scaling_normalization
- decision_trees: overfitting_underfitting
- random_forest: decision_trees, bias_variance
- boosting: decision_trees, gradient_descent
- xgboost: boosting
- lightgbm: boosting
- catboost: boosting, categorical_encoding

## Нейросети

- perceptron: linreg
- mlp: perceptron, logreg
- activation_functions: mlp
- loss_functions: mlp
- backpropagation: mlp, gradient_descent
- optimizers: backpropagation
- pytorch_full_guide: backpropagation, optimizers
- tensorflow_keras_full_guide: backpropagation, optimizers
- pytorch_lightning: pytorch_full_guide

## Компьютерное зрение

- cnn_basics: mlp
- pooling_layers: cnn_basics
- cnn_architectures: pooling_layers
- resnet: cnn_architectures
- transfer_learning_cnn: resnet
- data_augmentation: cnn_basics, overfitting_underfitting
- yolo: cnn_architectures
- object_detection: yolo
- image_segmentation: cnn_architectures
- 3d_computer_vision: cnn_architectures
- pose_estimation: object_detection
- neural_style_transfer: transfer_learning_cnn

## Последовательности, NLP и трансформеры

- rnn_basics: backpropagation
- lstm: rnn_basics
- gru: lstm
- word_embeddings: mlp
- attention_mechanism: rnn_basics, word_embeddings
- self_attention: attention_mechanism
- multi_head_attention: self_attention
- transformers: multi_head_attention
- bert_language_models: transformers
- gpt_architectures: transformers
- prompt_engineering: gpt_architectures
- vision_transformers_vit: transformers, cnn_basics
- tokenization: word_embeddings
- bert_finetuning: bert_language_models, tokenization
- nlp_tasks_ner_classification_summarization: bert_finetuning
- topic_modeling_lda_nmf: dimensionality_reduction

## Генеративные модели

- autoencoders: mlp
- vae: autoencoders
- gan: mlp, loss_functions
- diffusion_models: vae

## Временные ряды

- time_series: regression_metrics
- exponential_smoothing: time_series
- prophet_time_series: time_series
- rnn_lstm_time_series: lstm, time_series
- cnn_time_series_tcn: cnn_basics, time_series
- transformers_time_series: transformers, time_series

## Обучение с подкреплением

- q_learning_sarsa: reinforcement_learning_basics
- multi_armed_bandits: reinforcement_learning_basics
- dqn: q_learning_sarsa, mlp
- policy_gradient: reinforcement_learning_basics, backpropagation
- ppo_reinforcement_learning: policy_gradient

## Production и интерпретируемость

- hyperparameter_tuning: cross_validation
- bayesian_optimization: hyperparameter_tuning
- optuna_hyperopt: bayesian_optimization
- feature_importance: random_forest
- shap: feature_importance
- lime: feature_importance
- partial_dependence: feature_importance
- explainable_ai_xai: shap, lime
- production_ml_best_practices: cross_validation
- mlops_best_practices: production_ml_best_practices
- model_serialization: production_ml_best_practices
- ml_pipelines: preprocessing, cross_validation
- model_monitoring_ab_testing: production_ml_best_practices
- feature_stores: ml_pipelines
- cicd_ml: mlops_best_practices
- imbalanced_data: roc_auc
- concept_drift: model_monitoring_ab_testing
- fairness_ml: classification_metrics
- adversarial_robustness: cnn_basics, backpropagation
