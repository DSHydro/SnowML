# Using JupyterLab hosted on an EC2 Instance

The following instructions should help you spin up an EC2 instance, run JupyterLab on that instance, and access it from your local computer.

## Create EC2 Instance, download `.pem` file, alter Security Group
Create an EC2 Instance and make sure to take care of the following things:
1. Make sure to have the `.pem` file associated with the EC2 Instance. Create a new one or ask for it from the original creator of the file. Further, make sure to use the following command
2. on the file: `chmod 400 .pem`
3. Choose the `default` Security Group. This will allow for `ssh` connection from your IP address to the EC2 Instance.
4. Choose a large machine and crank up the amount of storage so that there is enough space for the data.
5. Go to EC2 dashboard, highlight the instance that was just created, go to the bottom and click the `Networking` tab. Note the `Public IPv4 address`. This will be used in a command later whenever you see `ip-address`.

## SSH to EC2 Instance with Local Port Forwarding
Open up a terminal and use the following command:
```
ssh -i <file>.pem -L localhost:8889:localhost:8889 ec2-user@<ip-address>
```

## Install Docker
Run the following commands while in the EC2 Instance via ssh:
```
sudo yum update -y
sudo yum install docker -y
sudo usermod -a -G docker ec2-user
sudo systemctl enable docker.service
sudo systemctl start docker.service
```
When all of the above commmands have been ran, exit out of the EC2 Instance (either type `exit` or close the terminal) and reconnect to it from the `ssh` command from above.

Type `docker info` to make sure docker is installed.

## Start up JupyterLab with Docker
Create a directory where you want all of your things to be saved and go into that directory. For example:
```
mkdir test
cd test
```
Now, run the following command:
```
docker run --rm -d -p 8889:8888 -v "$(pwd):/home/jovyan/work" quay.io/jupyter/base-notebook start-notebook.py --NotebookApp.token='token'
```

This does a few things:
1. Downloads the JupyterLab Docker Image from DockerHub (`quay.io/jupyter/base-notebook`)
2. Starts a container based off that image (`Docker run`)
3. Maps the local EC2 instance port `8889` to the Docker container port `8888` so that we can access the container from a browser (`-p 8889:8888`)
4. Mounts the current working directory to the container so that any files place in the current directory are both visible in the container and also any changes or new files made are saved if the container shuts down (`-v "$(pwd):/home/jovyan/work"`)
5. Runs a python script to start JupyterLab (`start-notebook.py`)
6. Creates a token that acts as a bit of a security measure (`--NotebookApp.token='token'`)

## Connect to JupyterLab locally
After everything above is complete, go to your favorite web browser and type in the following at the search bar:

```
https://localhost:8889/lab?token=token
```

## Success :)
