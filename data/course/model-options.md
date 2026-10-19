---
title: Model Options (Course Notes)
doc_id: model-options
course: AI Engineering
version: 2
updated: 2026-09-20
---
# Model Options (Course Notes)

## 1. Advantages of fine-tuning

Fine-tuning changes the model's weights with your own examples. Its advantages are a consistent style and format, shorter prompts, and better performance on a narrow task. It is a poor way to add new facts, because every update means training again.

## 2. Advantages of long context

Long-context models can read a whole small document collection in one prompt. The advantage is simplicity: no index, no chunking and no retrieval step, and the model can compare everything at once. The cost is that every question pays for every token of the corpus.

## 3. Advantages of local models

Running a free model on your own laptop keeps data private and costs nothing per question. The trade-off is lower quality than the largest hosted models and slower answers on a CPU.
