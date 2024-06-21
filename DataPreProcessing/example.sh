#!/bin/bash
USERNAME="viking"
ip="10.10.100.235"

function call()
{
    echo "Downloading Example from the CIMAT Server"
# get the ip direction of the local computer
    echo "USERNAME: $USERNAME"
    echo "IP: $ip"
    echo "The ip direction of the local computer is: $ip"
    scp -r ${USERNAME}@$ip:/home/viking/Storage/DanielVallejo/ThesisRealData/MSE_703_RelaxoAI-02 $1
}

call $1
