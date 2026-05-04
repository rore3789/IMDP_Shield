Once you have the git repo installed, you can set up a docker with the dependencies already installed using

'''bash
docker build -t julia-python-container .
'''

You can then run the docker container, providing it access to all files in the directory with

'''bash
docker run -it --mount type=bind,source="$(pwd)/src",target=/usr/src/SHIELD/ julia-python-container
'''
This enables the image to have access to any changes you make to the folder, such as adding user data or dynamics.

You can see an example use case by running
'''bash
./run_example.sh
'''
This reads from Setup/example.yaml for information and performs the entire shield synthesis process (learning -> abstraction -> shield).
In this case, it also samples dynamics as we have equations available for them.
In general, the user should provide data in Systems/<SYSTEM_NAME> using the same format as the example (training_data.pkl)


Note that auto_LiRPA is from Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond. NeurIPS 2020.
Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh.
It is used to generate a linear relaxation of the kernel deep kernel so GP bounding can be done.
The code is formatted to function with this particular version of auto_LiRPA.
