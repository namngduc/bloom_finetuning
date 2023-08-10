# Bloom fine-tuning for generate text base on alpaca [dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) 

The training and fine-tuning process in `Fine-tuning Bloom-1b1.ipynb` notebook. <br>
For run the application. Go to `bloom_service` folder

### Run app on local machine
1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html)
2. Create environtment
```bash
$ conda create -n bloom python==3.10
```
3. Install dependencies
```bash
$ pip install -r requirements.txt
```
4. Run app via command:
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
        "instruction": "Generate a list of ten items a person might need for a camping trip",
        "input": ""  
    },
    {
        "instruction": "What is the capital of France?",
        "input": ""
    }
]'
```
Result:
```bash
[
    {
        "instuction": "Generate a list of ten items a person might need for a camping trip",
        "input": "",
        "output": "1. Tent\n2. Sleeping bag\n3. Camping stove\n4. Flashlight\n5. Water bottle\n6. Camping chairs\n7. Camping blankets\n8. Cooking utensils\n9. Camping equipment\n10. Firewood. The list could go on and on. It is up to the person to decide what items they need for a camping trip. It is up to them to decide what they want to do and where they want to go. It is up to them to make the most of their camping trip. It is up to them to decide what they want to do and where they"
    },
    {
        "instuction": "What is the capital of France?",
        "input": "",
        "output": "Paris is the capital of France. It is located in the southwestern region of the country. It is the largest city in France and the second-largest city in Europe. It is also the seat of the French government and the capital of the European Union. It is also the seat of the European Court of Justice. Paris is also home to the Louvre Museum, the Eiffel Tower, and the Arc de Triomphe. It is also the birthplace of the French Revolution. Paris is also home to the Louvre Museum, the Eiffel Tower, and the Arc de Triomphe. It is also the birthplace of the French Revolution. Paris"
    }
]
```
