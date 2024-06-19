#!/bin/bash

# This script is used to run the bfcorr program on a set of files.

function correct()
{
    #Función de corrección del biasfield dentro de las imágenes de MR
    if [ $2 -eq 5 ]
    then
        mrconvert ${1}.nii.gz -coord 3 0 -axes 0,1,2,4 ${1}.nii.gz -force
    fi &&
    mrconvert ${1}.nii.gz ${1}_1.nii.gz -coord 3 0 -axes 0,1,2 -force &&
    N4BiasFieldCorrection -d 3 -i ${1}_1.nii.gz -o [${1}_n4.nii.gz,${1}_biasfield.nii.gz] -s 2 -v &&
    if [ -d ${1} ]
    then
        rm -r ${1}
    fi &&
    mkdir ${1} &&
    mkdir ${1}/biasfieldfiles &&
    cp ${1}.nii.gz ${1}/biasfieldfiles &&
    cp ${1}_1.nii.gz ${1}/biasfieldfiles &&
    mv ${1}_n4.nii.gz ${1}/biasfieldfiles  &&
    cp ${1}_biasfield.nii.gz ${1}/biasfieldfiles && 
    python3 biasfieldcorrector.py --path ${1}.nii.gz --bffile ${1}_biasfield.nii.gz
}

function run_all()
{
    correct $1 $2 &&
    python3 process.py --path ${1}.nii.gz --echos 32 --repetitions 16 &&
    #Primera parte de la corrección de sesgo de intensidad
    #Segunda parte, corrección de movimiento de los datos
	rm  ${1}_biasfield.nii.gz &&
	rm ${1}_1.nii.gz &&    
    cd ${1} &&
    for f in ${1}*
    do
            mrconvert -vox 8.0,1.0,1.0,1.0  $f v_${f} -force
    done &&
    for f in v_${1}*
    do
            mcflirt $f -out mc_${f} -report -verbose 1 -refvol 0 
    done && 
    for f in mc_v_${1}*
    do
            mrconvert -vox ${3},${4},${5},0  $f $f -force
    done &&
	mkdir -p toSignalDrift &&
	cp mc_v_${1}* ./toSignalDrift &&
    mkdir -p files &&
    for i in {0..15}
    do
            for f in mc_v*
            do
                    mrconvert $f -coord 3 0:$i ${i}_reps_${f} -force
            done
            for f in ${i}_reps_*
            do
                    mrmath $f mean avg_${f} -axis 3 -force
            done
            mrcat avg_${i}_reps* ./files/cat_mc_v_bf_reps_${i}_${1}.nii.gz -axis 3 -force             
    done && 
    rm *.nii.gz && 
    cd .. &&
    wait
}

run_all $1 $2 $3 $4 $5
