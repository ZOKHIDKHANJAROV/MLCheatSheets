#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Сериализация: pipeline, joblib/pickle, SHA-256 и проверка метаданных
"""

import pandas as pd
import numpy as np

import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['age', 'income', 'tenure', 'n_products']


def version(v, parts):
    return tuple(int(p) for p in str(v).split('.')[:parts])


def main():
    df = pd.read_csv('input.csv')
    env = json.load(open('service_env.json', encoding='utf-8'))
    train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
    X_tr, X_te = train[FEATURES], test[FEATURES]

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, train['target'])
    proba = pipe.predict_proba(X_te)[:, 1]
    rows = [('test_accuracy', pipe.score(X_te, test['target'])), ('proba_sum', proba.sum())]

    with tempfile.TemporaryDirectory() as d:
        jl, pk = Path(d) / 'model.joblib', Path(d) / 'model.pkl'
        joblib.dump(pipe, jl)
        with open(pk, 'wb') as f:
            pickle.dump(pipe, f)
        meta = {'sha256': hashlib.sha256(jl.read_bytes()).hexdigest(), 'features': FEATURES}

        # --- «в сервисе» ---
        sha_ok = hashlib.sha256(jl.read_bytes()).hexdigest() == meta['sha256']
        bad = Path(d) / 'tampered.joblib'
        shutil.copy(jl, bad)
        with open(bad, 'ab') as f:
            f.write(b'\x00')
        sha_bad = hashlib.sha256(bad.read_bytes()).hexdigest() == meta['sha256']

        loaded_jl = joblib.load(jl)
        with open(pk, 'rb') as f:
            loaded_pk = pickle.load(f)

    diff_jl = np.abs(loaded_jl.predict_proba(X_te)[:, 1] - proba).max()
    diff_pk = np.abs(loaded_pk.predict_proba(X_te)[:, 1] - proba).max()
    features_ok = list(loaded_jl.feature_names_in_) == env['features']
    # Ошибка «сохранили только модель»: последний шаг на сырых признаках
    changed = int((pipe[-1].predict(X_te.to_numpy()) != pipe.predict(X_te)).sum())
    rows += [('max_diff_joblib', diff_jl), ('max_diff_pickle', diff_pk), ('sha_match', sha_ok),
             ('sha_tampered_match', sha_bad), ('features_match', features_ok),
             ('n_changed_without_scaler', changed)]

    reg = pd.read_csv('registry.csv')
    for r in reg.itertuples():
        ok = (version(r.sklearn_version, 2) == version(env['sklearn_version'], 2)
              and version(r.numpy_version, 1) == version(env['numpy_version'], 1)
              and r.features.split('|') == env['features'])
        rows.append((f'compat_{r.model_id}', ok))

    result = pd.DataFrame(rows, columns=['check', 'value'])
    result['value'] = result['value'].astype(float).round(4)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
