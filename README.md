# Development of Intelligent Systems

[![Video](https://img.youtube.com/vi/kIe1qp8iarY/maxresdefault.jpg)](https://youtu.be/kIe1qp8iarY)

In this report we describe the methods, implementation and integration and the results achieved solving
tasks in the course Development of Intelligent Systems in the academic year 2021/2022. We also state
how the work was divided and give some final remarks on our work at the end of the report.
The project includes designing and implementing a system in ROS that completes the following task successfully in a simulator: the robot should autonomously explore the given fenced area. While exploring, the
robot should:
- find all persons in this area and recognize them (the faces are attached to the inner walls),
- find all restaurants (different coloured cylinders),
- recognize the food each of these restaurants serve (a circular image on top of the cylinder),
- find a green ring that marks a parking spot.
After finding all of these and saving their positions, the robot should go to the parking spot and accept
orders from the link given in the QR code on the wall next to the parking spot. The robot should then
plan the delivery of each of the orders, i.e. go to the restaurant that serves the type of food in the order
and then go to the person in the order. When delivering the order to the persons, the robot should have
a short dialogue with them.

To better present the implemented project, we made a recording of how the robot works in a given
simulated environment. The video is available at the following link: https://youtu.be/kIe1qp8iarY.
