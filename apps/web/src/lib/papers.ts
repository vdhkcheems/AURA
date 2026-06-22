export type SupportedPaper = {
  id: string;
  title: string;
  year: number;
  topics: string[];
};

export const supportedPapers: SupportedPaper[] = [
  { id: "attention-is-all-you-need", title: "Attention Is All You Need", year: 2017, topics: ["transformers", "attention"] },
  { id: "bert", title: "BERT", year: 2018, topics: ["transformers", "language-modeling"] },
  { id: "gpt-3", title: "Language Models are Few-Shot Learners", year: 2020, topics: ["language-modeling", "scaling"] },
  { id: "batch-normalization", title: "Batch Normalization", year: 2015, topics: ["training", "normalization"] },
  { id: "dropout", title: "Dropout", year: 2012, topics: ["regularization", "training"] },
  { id: "resnet", title: "Deep Residual Learning", year: 2015, topics: ["computer-vision", "residual-networks"] },
  { id: "clip", title: "CLIP", year: 2021, topics: ["multimodal-learning", "vision-language"] },
  { id: "ddpm", title: "Denoising Diffusion Probabilistic Models", year: 2020, topics: ["diffusion-models", "generative-modeling"] },
  { id: "rag", title: "Retrieval-Augmented Generation", year: 2020, topics: ["retrieval", "language-modeling"] },
  { id: "dense-passage-retrieval", title: "Dense Passage Retrieval", year: 2020, topics: ["retrieval", "question-answering"] },
];
