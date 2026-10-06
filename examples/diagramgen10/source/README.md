---
license: apache-2.0
task_categories:
- text2text-generation
- visual-question-answering
- image-to-text
language:
- en
configs:
- config_name: DiagramCoding
  data_files:
  - split: test
    path: DiagramCoding.json
- config_name: DiagramEditing
  data_files:
  - split: test
    path: DiagramEditing.json
- config_name: DiagramGeneration
  data_files:
  - split: test
    path: DiagramGeneration.json
    
---


[📑paper link](https://arxiv.org/abs/2411.11916)

## Dataset Card: DiagramAgent/DiagramGenBenchmark

### 1. Overview
**DiagramAgent/DiagramGenBenchmark** is a comprehensive benchmark designed for evaluating text-to-diagram generation and editing tasks. It provides a diverse set of diagram types alongside corresponding textual descriptions and code representations, aiming to facilitate research in generating structured visual content from natural language inputs.

### 2. Dataset Description
- **Objective**:  
  To transform textual instructions into structured, logically coherent diagrams.  
- **Content**:  
  The dataset includes a wide range of diagram types:
  - **Model Architecture Diagrams**
  - **Flowcharts**
  - **Line Charts**
  - **Directed Graphs**
  - **Undirected Graphs**
  - **Tables**
  - **Bar Charts**
  - **Mind Maps**

- **Data Format**:  
  Each sample typically contains:
  - A user instruction or query describing the diagram.
  - The corresponding diagram code (written primarily in LaTeX or DOT) that can be compiled into a visual diagram.

### 3. Data Sources
- The dataset aggregates samples from multiple public resources:
  - HuggingFace’s VGQA dataset
  - Datikz and Datikz-v2 datasets
  - Open-source repositories on GitHub and Overleaf  
- **Licensing**:  
  The sources are licensed under CC BY 4.0 or MIT, ensuring open access while respecting original content rights.

### 4. Citation

If you find our work helpful, feel free to give us a cite.


```
@inproceedings{wei2024wordsstructuredvisualsbenchmark,
  title={From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing},
  author={Jingxuan Wei and Cheng Tan and Qi Chen and Gaowei Wu and Siyuan Li and Zhangyang Gao and Linzhuang Sun and Bihui Yu and Ruifeng Guo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```