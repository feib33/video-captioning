This project is modified based on [TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval](https://arxiv.org/abs/2001.09099).

The result is improved by processing video features and subtitle features separately and integrating before inputting into the original decoder 

### TVC Dataset and Task

The TVC dataset can be refered by contacting original auther @jielei

### Method: MultiModal Transformer (MMT)

## Acknowledgement

Thanks for original paper author @jielei who released the source code.

```
@inproceedings{lei2020tvr,
  title={TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval},
  author={Lei, Jie and Yu, Licheng and Berg, Tamara L and Bansal, Mohit},
  booktitle={ECCV},
  year={2020}
}
```

## Acknowledgement from original author
This research is supported by grants and awards from NSF, DARPA, ARO and Google.

This code borrowed components from the following projects: 
[recurrent-transformer](https://github.com/jayleicn/recurrent-transformer),
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), 
[transformers](https://github.com/huggingface/transformers),
[coco-caption](https://github.com/tylin/coco-caption),
we thank the authors for open-sourcing these great projects! 

