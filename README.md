Hierarchical Time-aware Approach for Video Summarization
=====
Code for our Bracis paper "Hierarchical Time-aware Approach for Video Summarization" Enhanced by Leonardo Vilela Cardoso, Silvio Jamil F. Guimarães and Zenilton K. G. Patrocínio Jr,

Video summarization consists of generating a concise video representation that captures all its meaningful information. However, conventional summarization techniques often fall short of capturing all the significant events in a video due to their inability to incorporate the hierarchical structure of the video content. This work proposes an unsupervised method, named <b>Hie</b>rarchical <b>T</b>ime-<b>a</b>ware <b>Summ</b>arizer -- HieTaSumm, that uses a hierarchical approach for that task. In this regard, hierarchical strategies for video summarization have emerged as a promising solution, in which video content is modeled as a graph to identify keyframes that represent the most relevant information. This approach enables the extraction of the frames that convey the central message of the video, resulting in a more effective and precise summary. Experimental results indicate that the proposed approach has great potential. Specifically, it seems to enhance coherence among different video segments, reducing frame redundancy in the generated summaries, and enhancing the diversity of selected keyframes.

## Getting started
### Prerequisites
0. Clone this repository
```
# no need to add --recursive as all dependencies are copied into this repo.
git clone https://github.com/IMScience-PPGINF-PucMinas/Video-Summarization.git
cd HieTaSumm/
```

1. Prepare feature files

we applied the proposed method to the same collections of videos from the OpenVideo dataset (referred to as the VSUMM dataset available <a href='https://sites.google.com/site/vsummsite/download'>here </a>). This dataset contains 50 videos of different genres. All videos are in MPEG-1 format (30 fps, 352 X 240 pixels). The genres are distributed into documentary, educational, ephemeral, historical, and lecture. The time duration of each video varies from 01 to 04 minutes. The process of creating of user summary consists of the collaboration of 50 different persons. Each user is dealing with the task of choosing the keyframes for 5 videos. Thus, 250 were created for the dataset each video has 05 different user summaries generated manually. And, as a way to pre-process the video dataset %, and enable the creation of the frames dataset, we extracted 04 fps from all videos.

2. Install dependencies
- Python 3.6
- PyTorch 1.1.0
- nltk
- easydict
- tqdm
- opencv

### Training and Inference
All codes needed to run HieTaSumm is available on HieTaSumm.py.


1. To run the HieTaSumm:

The general training command is:
```
python HieTaSumm.py "path_to_videos" "path_to_frames_output" "path_to_Users_Summary" "rate_int" "time_max_connection"
```

## Citations
If you find this code useful for your research, consider cite our paper:
```
@INPROCEEDINGS{cardoso2021summarization,
    AUTHOR="Leonardo Vilela Cardoso and Zenilton Kleber Patrocínio Jr and Silvio Guimarães",
    TITLE="Hierarchical Time-aware Approach for Video Summarization",
    BOOKTITLE="BRACIS 2023",
    ADDRESS="Belo Horizonte, MG, Brazil",
    DAYS="25-29",
    MONTH="sep",
    YEAR="2023",
}
```

## Contact
Leonardo Vilela Cardoso with this e-mail: lvcardoso@sga.pucminas.br
