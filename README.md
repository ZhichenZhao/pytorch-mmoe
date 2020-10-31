# pytorch-mmoe

This project is a re-implementation of MMoE [Modeling Task Relationships in Multi-task Learning with
Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007). The reference code is the keras version of MMoE: [keras-version](https://github.com/drawbridge/keras-mmoe)

## How to use
mmoe = MMoEModule(input_size, units, num_experts, num_tasks)
output = mmoe(input)



