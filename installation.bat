cd /d D:
git clone https://github.com/prafful91/Image_comp_dlib.git
cd /d D:/Image_comp_dlib
pip install virtualenv
virtualenv venv --python 3.10.9
call venv\Scripts\activate
pip install cmake
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
pip install -r requirements.txt
python first_run.py
uvicorn main_deepface:app --reload
