# Cleanse
Cleanse is a hand hygiene compliance software that cultivates healthy hand hygiene habits through proper guidance and reminders. Through this, we aim to improve and raise awareness about the importance of hand washing, especially in this time of crisis, where the disease is spreading like wildfire. 

The Cleanse prototype utilizes multiple machine learning algorithms and models to track and guide users through various handwashing actions. We created a custom data set by capturing and feeding images of the correct hand poses for the 7 different handwashing steps. These captured data are collected and normalized to allow the model to recognize variations in hand sizes. The data is then fed into a neural network where we trained the model before deployment. Through bluetooth, the micro bit receives signals through the cloud, communicating with and activating the lighting system, signaling to users when each step is completed.

Through this project, we learned to use different libraries and frameworks like p5js, ml5, tensor flow, mediapipe and the microbit. One of the biggest difficulties that we faced was trying to train an accurate hand washing model for Cleanse. We had to take special care of various angles and hand sizes during the data collection stage which could affect the accuracy of the results.

With Cleanse, we hope to empower these frontline workers with the necessary knowledge and technology, to protect them against the potential threats of COVID-19 as first responders in this pandemic.
