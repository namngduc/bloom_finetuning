### Run app on local environtment
1. Install dependencies
```bash
$ pip install -r requirements.txt
```
2. Run app via command:
```bash
$ python app.py
```
App will run on host: `http://0.0.0.0:7860`

### Run app via docker
1. Build docker image
```bash
$ docker build -t bloom_zalo .                                 
```
2. Run docker container:
```bash
$ docker run --rm -p 7860:7860 --name bloom_service bloom_zalo
```

### Demo
![demo](./images/Screenshot%202023-07-21%20at%2014.22.01.png)  