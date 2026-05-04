This repository is an end to end framework on data-driven shielding. Given a dataset, we first learn dynamics using Deep Kernel Learning. 
Then this model is abstracted into a finite state model, the Interval Markov Decision Process (IMDP).
The product IMDP is constructed using a DFA of the safety specification and the shield synthesis algorithm is run on this product.
The algorithm iteratively removes actions from the product IMDP until until a fixed point is reach with value iteration where no more unsafe actions are identified.
The result is a set of safe actions, their values, and the associated product states for those actions.

Details on installation and use are in the README in the src folder.
