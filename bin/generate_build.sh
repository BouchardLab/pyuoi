eval "$(conda shell.bash hook)"
mkdir dist
for py in 3.5 3.6 3.7; do
  git clone https://github.com/BouchardLab/pyuoi.git
  cd pyuoi
  conda create -y -n temp_build_env python=$py
  conda activate temp_build_env
  conda install -y numpy cython
  pip install setuptools wheel
  python setup.py sdist bdist_wheel
  conda deactivate
  conda remove -y -n temp_build_env --all
  mv dist/* ../dist/.
  cd ..
  rm -rf pyuoi
done
