#!/usr/bin/env python3

import requests
import pytest
import os

# Set up environment variables
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_OWNER = 'jklust'
REPO_NAME = 'gamdpy'
BRANCH_NAME = 'master'

def get_latest_commit():
    url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH_NAME}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['sha']

def set_commit_status(commit_sha, state, description, target_url):
    url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/statuses/{commit_sha}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    data = {
        'state': state,
        'description': description,
        'context': 'CI Server',
        'target_url': target_url
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

def run_tests():
    # Run pytest and capture the results
    result = pytest.main(['-x', '--disable-warnings']) # --run-slow

    # Check the result code
    if result == 0:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed.")
        return False

def main():
    commit_sha = get_latest_commit()
    target_url="https://dirac.ruc.dk/..." # report from the pytest
    set_commit_status(commit_sha, 'pending', 'Running tests...', target_url)

    tests_passed = run_tests()

    if tests_passed:
        set_commit_status(commit_sha, 'success', 'All tests passed!', target_url)
    else:
        set_commit_status(commit_sha, 'failure', 'Tests failed.', target_url)

if __name__ == '__main__':
    main()
