python3 -m venv sat_relu_env
source sat_relu_env/bin/activate
pip install numpy
pip install torch
pip install onnx
python3 generate_properties.py 42
deactivate
