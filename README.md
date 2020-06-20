# TextClassification_GPT
An implementation of GPT to do TextClassification on Patent Data

For a complete PyTorch implementation of GPT, please refer to:
https://github.com/huggingface/pytorch-openai-transformer-lm

The original paper can be found here: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

I added some comments along with the codes. If you find it's hard to digest the huggingface's original codes, I hope this would be helpful.

The data can be download from USPTO for free. For sensitivity reasons, I only updated a demo dataset for reference.

I used 20000+ patent abstracts in my implementation, and the test accuracy can achieve 74% or so within four epochs (fine tuning). Beyond 5 epochs, GPT would overfit the data (e.g., 100% training accuracy for 9 epochs).
