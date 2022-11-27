# DLS-Project
## Resources

* Optimizing Sparse Matrix Multiplications for
Graph Neural Networks

https://lcpc2021.github.io/pre_workshop_papers/Qiu_lcpc21.pdf

* CUDA cuSparse Documentation

https://developer.nvidia.com/cusparse

* PyTorch Sparse Documentation
https://pytorch.org/docs/stable/sparse.html

* GNN application of SparseTensor in PyG
https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html

# Proposal: 
In this project, we plan to implement support of sparse tensors in Needle. Sparse Tensors (Matrices, Vectors etc.) are tensors that have most entries being zero. For these tensors, more efficient storage formats exist that allows storing these tensors in compression. Such compression can potentially reduce memory consumption, improve computation speed and enable larger tensor sizes when the deep learning models are inherently sparse. We plan to implement the SparseTensor interface in Needle as well as the associated unary and binary operations and develop and conduct unit tests for our implementation. We also plan to explore acceleration methods using CPU and GPU. Finally, our implementation can be validated on actual deep learning models. We plan to use the SparseTensor interface in PyTorch as a reference to guide our own implementation. Wherever applicable, we may leverage CUDA library (e.g., cuSparse) for GPU acceleration.
