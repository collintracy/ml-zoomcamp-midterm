#!/usr/bin/env python
# Send one animal to the running service and print the prediction.
# Usage: python predict-test.py [url]

import sys

import requests


url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:9696/predict'

animal = {
    'animal_type': 'Dog',
    'intake_type': 'Stray',
    'intake_condition': 'Normal',
    'sex_upon_intake': 'Intact Male',
    'age_upon_intake': '2 years',
    'breed': 'Labrador Retriever Mix',
    'color': 'Black/White',
    'intake_datetime': '2025-03-01T10:00:00',
    'prior_stays': 0,
}

response = requests.post(url, json=animal).json()
print(response)

if response['adopted']:
    print('likely to be adopted')
else:
    print('at risk of not being adopted: flag for extra adoption outreach')
