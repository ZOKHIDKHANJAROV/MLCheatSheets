#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Валидатор вызовов инструментов по JSON-схеме
"""

import pandas as pd
import numpy as np

import json

STATUSES = ['ok', 'unknown_tool', 'bad_json', 'missing_field', 'extra_field', 'bad_type', 'bad_enum',
            'out_of_range']


def type_ok(value, expected):
    if expected == 'string':
        return isinstance(value, str)
    if expected == 'boolean':
        return isinstance(value, bool)
    if isinstance(value, bool):                      # bool — подкласс int, но числом не считается
        return False
    if expected == 'integer':
        return isinstance(value, int)
    if expected == 'number':
        return isinstance(value, (int, float))
    return False


def validate(tool, arguments_text, schemas):
    if tool not in schemas:
        return 'unknown_tool'
    try:
        args = json.loads(arguments_text)
    except ValueError:
        return 'bad_json'
    if not isinstance(args, dict):
        return 'bad_json'
    schema = schemas[tool]
    props = schema['properties']
    if any(name not in args for name in schema['required']):
        return 'missing_field'
    if any(name not in props for name in args):
        return 'extra_field'
    if any(not type_ok(value, props[name]['type']) for name, value in args.items()):
        return 'bad_type'
    if any('enum' in props[name] and value not in props[name]['enum'] for name, value in args.items()):
        return 'bad_enum'
    for name, value in args.items():
        spec = props[name]
        if 'minimum' in spec and value < spec['minimum']:
            return 'out_of_range'
        if 'maximum' in spec and value > spec['maximum']:
            return 'out_of_range'
    return 'ok'


def main():
    calls = pd.read_csv('input.csv', dtype=str, keep_default_na=False)
    with open('schemas.json', encoding='utf-8') as f:
        schemas = json.load(f)
    calls['status'] = [validate(t, a, schemas) for t, a in zip(calls['tool'], calls['arguments'])]
    counts = calls['status'].value_counts().reindex(STATUSES, fill_value=0)
    result = pd.DataFrame({'status': STATUSES, 'count': counts.to_numpy()})
    result['share'] = (result['count'] / len(calls)).round(4)
    print(f"корректных вызовов: {int(counts['ok'])} из {len(calls)}")
    return result


if __name__ == "__main__":
    result = main()
    output_file = 'solution_output.csv'  # answer.csv — эталон, его не перезаписываем
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результат сохранён в {output_file}")
    print(result.head(10))
