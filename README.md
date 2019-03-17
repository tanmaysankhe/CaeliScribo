# CaeliScribo

Simulation of Air Writing Softwares for Smart Wearables.

![image](https://github.com/tanmaysankhe/CaeliScribo/blob/master/screeenshots/ss1.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
Tensorflow 1.12
```
```
Tensorflow Object Detection 
```
```
Flask 
```
```
Protobuf
```

### Installing Prerequisites

A step by step series of examples that tell you how to get a development env running


Install TensorFlow version 1.12

```
pip install tensorflow==1.12
```


Install TensorFlow GPU version 1.12
```
pip install tensorflow-gpu==1.12
```


Clone tensorflow/models
```
git clone https://github.com/tensorflow/models.git
```


Install Protobuf
```
Make sure you grab the latest version
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip

Unzip
unzip protoc-3.3.0-linux-x86_64.zip -d protoc3

Move protoc to /usr/local/bin/
sudo mv protoc3/bin/* /usr/local/bin/

 Move protoc3/include to /usr/local/include/
sudo mv protoc3/include/* /usr/local/include/

Optional: change owner
sudo chown $USER /usr/local/bin/protoc
sudo chown -R $USER /usr/local/include/google
```



Install Flask
```
pip install flask
```


End with an example of getting some data out of the system or using it for a little demo

## Running the system

Clone our repository
```
git clone https://github.com/tanmaysankhe/CaeliScribo
```

Copy the contents of repository in
```
models/research/objectdetection
```

Finally, open terminal in objectdetection folder and run
```
python main.py
```

## Authors

* **Tanmay Sankhe** - [PurpleBooth](https://github.com/tanmaysankhe)


* **Mumbaikar007** - [PurpleBooth](https://github.com/Mumbaikar007)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Sendex Tutorials](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/)
* [Pranav Mistry Ted Talk - Sixth Sense](https://www.ted.com/talks/pranav_mistry_the_thrilling_potential_of_sixthsense_technology?language=en)
* StackOverflow

