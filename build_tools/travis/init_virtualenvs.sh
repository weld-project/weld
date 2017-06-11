 #!/bin/bash

PYTHON_VERSIONS=$@

pip install -U virtualenv
mkdir -p .virtualenv
export VENV_HOME=`pwd`/.virtualenv

pushd python
for v in $PYTHON_VERSIONS; do
  virtualenv "$VENV_HOME/python$v" --python="python$v"
  source "$VENV_HOME/python$v/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install --only-binary=numpy,pandas -r requirements.txt
  deactivate
done
popd