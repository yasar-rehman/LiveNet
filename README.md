# LiveNet

The files are currently in the raw format. 
# Abstract
Performance of face liveness detection algorithms in cross-database face liveness detection tests is one of the key issues in face-biometric based systems. Recently, Convolution Neural Networks (CNN) classifiers have shown remarkable performance in intra-database face liveness detection tests. However, a little effort has been made to improve the generalization capability of CNN classifiers for cross-database and unconstrained face liveness detection tests. In this paper, we propose an efficient strategy for training deep CNN classifiers for face liveness detection task. We utilize continuous data-randomization (like bootstrapping) in the form of small mini-batches during training CNN classifiers on small scale face anti-spoofing database. Experimental results revealed that the proposed approach reduces the training time by 18.39%, while significantly lowering the HTER by 8.28% and 14.14% in cross-database tests on CASIA-FASD and Replay-Attack database respectively as compared to state-of-the-art approaches. Additionally, the proposed approach achieves satisfactory results on intra-database and cross-database face liveness detection tests, claiming a good generality over other state-of-the-art face anti-spoofing approaches.



# Citation:

If you find our work useful, please cite it as follows:
```
@article{rehman2018livenet,
  title={LiveNet: Improving features generalization for face liveness detection using convolution neural networks},
  author={Rehman, Yasar Abbas Ur and Po, Lai Man and Liu, Mengyang},
  journal={Expert Systems with Applications},
  volume={108},
  pages={159--169},
  year={2018},
  publisher={Elsevier}
}
```
