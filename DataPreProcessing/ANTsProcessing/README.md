This folder contains two register pipelines, both work with ANTsPY and use a BOLDRigid transformation for motion correction. The algorithms use MPPCA for denoising each one of the acquisitions. The two algorithms are described below

* The first pipeline recieves the separated echoes into a different volumes of size $(x,y,z,echos)$ and separates the files into volumes of size $(x,y,z,echos)$ for the $n$ acquisitions. To run this code use ``` python register_pipeline_se.py --path (path to folder with acquisitions) --n_rep 16 --save_path (save path)``` 
* The second pipeline recieves a folder containing the different acquisitions of size $(x,y,z,echos)$ for registration, denoising and motion correction.
