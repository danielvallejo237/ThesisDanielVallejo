#!/bin/bash
USERNAME="viking"
echo "Downloading Example from the CIMAT Server"
# get the ip direction of the local computer
ip="10.10.100.235"
echo "USERNAME: $USERNAME"
echo "IP: $ip"
echo "The ip direction of the local computer is: $ip"
scp -r ${USERNAME}@$ip:/home/viking/Storage/DanielVallejo/ThesisRealData/MSE_703_RelaxoAI-02 ./