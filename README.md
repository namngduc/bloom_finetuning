# Bloom fine-tuning for generate text base on alpaca [dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) 

The training and fine-tuning process in `Fine-tuning Bloom-1b1.ipynb` notebook. You can try to optimize the model further by yourself. <br>
For run the application. Go to `bloom_service` folder

### Run app on local machine
1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html)
2. Create environtment
```bash
$ conda create -n bloom python==3.10
```
3. Activate environtment
```bash
$ conda activate bloom
```
4. Install dependencies
```bash
$ pip install -r requirements.txt
```
5. Run app via command:
```bash
$ chmod +x start_app.sh
```
```bash
$ ./start_app.sh
```
You need at least a GPU with 6GB VRAM for inference. The service was optimized with inference via batch. Currently, I set default value as `2`.
You can change the batch size by export `BATCH_SIZE` environtment variable.
```bash
$ export BATCH_SIZE=new_value
```
The API endpoint: `http://0.0.0.0:8080/bloom_generate` method: `POST` <br>
For the API documentation: `http://0.0.0.0:8080/docs`

### Run app via docker
1. Build docker image
```bash
$ docker build -t bloom_text .                                 
```
2. Run docker container:
```bash
$ docker run --rm -p 8080:8080 --name bloom_service bloom_text
```
You can also change BATCH_SIZE with docker command:
```bash
$ docker run --rm -p 8080:8080 -e BATCH_SIZE=new_value --name bloom_service bloom_text
```
The Dockerfile was only written for use on CPU. If you want to run service with Docker and GPU, you need install [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) and re-write the Dockerfile.

### Testing
You can use [POSTMAN](https://www.postman.com/downloads/) or use cURL:
```bash
curl --location --request POST 'http://0.0.0.0:8080/bloom_generate' \
--header 'Content-Type: application/json' \
--data-raw '[
    {
        "instruction": "What is the main idea of this text?",
        "input": "The robotics industry has become one of the fastest growing sectors in today’s economy. Robotics technology is being applied in various industries and applications, such as manufacturing, healthcare, and defense."  
    },
    {
        "instruction": "Describe in a few words what makes a good leader.",
        "input": ""
    }
]'
```
Result:
```bash
[
    {
        "instuction": "What is the main idea of this text?",
        "input": "The robotics industry has become one of the fastest growing sectors in today’s economy. Robotics technology is being applied in various industries and applications, such as manufacturing, healthcare, and defense.",
        "output": "The main idea of this text is that robotics technology is being used in various industries and applications, such as manufacturing, healthcare, and defense. It is being used to automate processes, increase efficiency, and reduce costs. Additionally, robotics technology is being used to improve safety and security, as well as to provide more personalized services. Overall, robotics technology is becoming increasingly important in the modern world. It is being used to revolutionize the way we work and live. It is a powerful"
    },
    {
        "instuction": "Describe in a few words what makes a good leader.",
        "input": "",
        "output": "A good leader is someone who is able to motivate and inspire others to achieve their goals. They are also able to build trust and relationships with their team members, and are able to communicate effectively. A good leader also has the ability to listen and understand their team members' needs, and to be able to provide constructive feedback. They are also able to build relationships with their stakeholders, and to be able to communicate effectively with them. A good leader is also able to be flexible and willing to"
    }
]
```
