cd meshpy
python setup.py develop
cd ..
cd dex-net
python setup.py develop
cd ..
pip install -r requirements.txt
cd cython-eval-new
make
mv *.so ..
