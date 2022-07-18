## campfire on zhang-testing 

High volume bleeding-edge branch for testing and remote deployment. Some vars hardcoded. NOT A PORTABLE IMPLEMENTATION.

Commits should be Blackened for consistency in code review and smaller diffs. This is set up in a pre-commit hook. To use, run:

```
git fetch --all
git pull

pip install black
pip install pre-commit
pre-commit install
```

This will set up Black to track and autorun over all future commits. If desired, OSX users can run `brew install pre-commit` instead.

