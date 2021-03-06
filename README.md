
# notebook
This is used to test the concept and model. It's often used by data scientists before 
handing you their code.

virtualenv -p /usr/local/bin/python3 env
source bin/activate

pip3 install jupyter
pip3 install pillow
pip3 install pandas
pip3 install tf-explain
pip3 install tensorflow
pip3 install -U matplotlib

jupyter notebook

# dev-server
This is the dev server for flask... not a production server.

Note: Make sure that you don't have an virtualenv loaded into the shell you're using... ie, automatically from vs code.

docker build -t image-api:latest .
docker run -d -p 5000:5000 image-api:latest
cd /Users/wesleyreisz/learn/image_classification/tests
curl -X POST -F image=@knights.jpg 'http://localhost:5000/predict' | jq

NOTE: if you can curl the result inside the container, but not outside you need to tell flask that it can accept requests
on other ips besides localhost: `app.run(host='0.0.0.0')`


# prod-server
Setup docker with nginx and gunicorn (Green Unicorn: Python WSGI HTTP Server for UNIX) server for flask
I also initially set this up with wsgi but ran into some issues. Tried Green Unicorn as the next recommended 
approach. It might make sense to try to run this with Django rather than Flask.

Download models and serve them via proxy folder.
example models: https://github.com/fchollet/deep-learning-models/releases 

# module
Project loads into the environment. However, gunicorn booting working is an infinite loop. Not sure why...