import torch
import random
import numpy as np
from models import get_model_class
from args_demo import parse_args_func
from transformers import BertTokenizer, BertModel
from ClassifierModel import BertClassifier, classifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args_func()

    if args.seed is not None:
        set_seed(args.seed)

    _, tokenizer_class, model_class = get_model_class(args.model_type)

    tokenizer = tokenizer_class.from_pretrained(args.model_dir)
    

    if args.model_half: 
        if 'llama' in args.model_type.lower():
            model = model_class.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map="auto", use_cache=args.use_cache, attn_implementation=args.attn_implementation).eval()
        else:
            model = model_class.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", use_cache=args.use_cache, attn_implementation=args.attn_implementation).eval()
    
    
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    bert_model = BertModel.from_pretrained(args.bert_model_dir, device_map="auto")

    bert_classifier = BertClassifier(bert_model).to(model.device)
    bert_classifier.load_state_dict(torch.load(args.classifier_dir))
    bert_classifier.eval()

    template = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    
    context = "Introduction\nIn the age of information dissemination without quality control, it has enabled malicious users to spread misinformation via social media and aim individual users with propaganda campaigns to achieve political and financial gains as well as advance a specific agenda. Often disinformation is complied in the two major forms: fake news and propaganda, where they differ in the sense that the propaganda is possibly built upon true information (e.g., biased, loaded language, repetition, etc.).\nPrior works BIBREF0, BIBREF1, BIBREF2 in detecting propaganda have focused primarily at document level, typically labeling all articles from a propagandistic news outlet as propaganda and thus, often non-propagandistic articles from the outlet are mislabeled. To this end, EMNLP19DaSanMartino focuses on analyzing the use of propaganda and detecting specific propagandistic techniques in news articles at sentence and fragment level, respectively and thus, promotes explainable AI. For instance, the following text is a propaganda of type `slogan'.\nTrump tweeted: $\\underbrace{\\text{`}`{\\texttt {BUILD THE WALL!}\"}}_{\\text{slogan}}$\nShared Task: This work addresses the two tasks in propaganda detection BIBREF3 of different granularities: (1) Sentence-level Classification (SLC), a binary classification that predicts whether a sentence contains at least one propaganda technique, and (2) Fragment-level Classification (FLC), a token-level (multi-label) classification that identifies both the spans and the type of propaganda technique(s).\nContributions: (1) To address SLC, we design an ensemble of different classifiers based on Logistic Regression, CNN and BERT, and leverage transfer learning benefits using the pre-trained embeddings/models from FastText and BERT. We also employed different features such as linguistic (sentiment, readability, emotion, part-of-speech and named entity tags, etc.), layout, topics, etc. (2) To address FLC, we design a multi-task neural sequence tagger based on LSTM-CRF and linguistic features to jointly detect propagandistic fragments and its type. Moreover, we investigate performing FLC and SLC jointly in a multi-granularity network based on LSTM-CRF and BERT. (3) Our system (MIC-CIS) is ranked 3rd (out of 12 participants) and 4th (out of 25 participants) in FLC and SLC tasks, respectively.\nSystem Description ::: Linguistic, Layout and Topical Features\nSome of the propaganda techniques BIBREF3 involve word and phrases that express strong emotional implications, exaggeration, minimization, doubt, national feeling, labeling , stereotyping, etc. This inspires us in extracting different features (Table TABREF1) including the complexity of text, sentiment, emotion, lexical (POS, NER, etc.), layout, etc. To further investigate, we use topical features (e.g., document-topic proportion) BIBREF4, BIBREF5, BIBREF6 at sentence and document levels in order to determine irrelevant themes, if introduced to the issue being discussed (e.g., Red Herring).\nFor word and sentence representations, we use pre-trained vectors from FastText BIBREF7 and BERT BIBREF8.\nSystem Description ::: Sentence-level Propaganda Detection\nFigure FIGREF2 (left) describes the three components of our system for SLC task: features, classifiers and ensemble. The arrows from features-to-classifier indicate that we investigate linguistic, layout and topical features in the two binary classifiers: LogisticRegression and CNN. For CNN, we follow the architecture of DBLP:conf/emnlp/Kim14 for sentence-level classification, initializing the word vectors by FastText or BERT. We concatenate features in the last hidden layer before classification.\nOne of our strong classifiers includes BERT that has achieved state-of-the-art performance on multiple NLP benchmarks. Following DBLP:conf/naacl/DevlinCLT19, we fine-tune BERT for binary classification, initializing with a pre-trained model (i.e., BERT-base, Cased). Additionally, we apply a decision function such that a sentence is tagged as propaganda if prediction probability of the classifier is greater than a threshold ($\\tau $). We relax the binary decision boundary to boost recall, similar to pankajgupta:CrossRE2019.\nEnsemble of Logistic Regression, CNN and BERT: In the final component, we collect predictions (i.e., propaganda label) for each sentence from the three ($\\mathcal {M}=3$) classifiers and thus, obtain $\\mathcal {M}$ number of predictions for each sentence. We explore two ensemble strategies (Table TABREF1): majority-voting and relax-voting to boost precision and recall, respectively.\nSystem Description ::: Fragment-level Propaganda Detection\nFigure FIGREF2 (right) describes our system for FLC task, where we design sequence taggers BIBREF9, BIBREF10 in three modes: (1) LSTM-CRF BIBREF11 with word embeddings ($w\\_e$) and character embeddings $c\\_e$, token-level features ($t\\_f$) such as polarity, POS, NER, etc. (2) LSTM-CRF+Multi-grain that jointly performs FLC and SLC with FastTextWordEmb and BERTSentEmb, respectively. Here, we add binary sentence classification loss to sequence tagging weighted by a factor of $\\alpha $. (3) LSTM-CRF+Multi-task that performs propagandistic span/fragment detection (PFD) and FLC (fragment detection + 19-way classification).\nEnsemble of Multi-grain, Multi-task LSTM-CRF with BERT: Here, we build an ensemble by considering propagandistic fragments (and its type) from each of the sequence taggers. In doing so, we first perform majority voting at the fragment level for the fragment where their spans exactly overlap. In case of non-overlapping fragments, we consider all. However, when the spans overlap (though with the same label), we consider the fragment with the largest span.\nExperiments and Evaluation\nData: While the SLC task is binary, the FLC consists of 18 propaganda techniques BIBREF3. We split (80-20%) the annotated corpus into 5-folds and 3-folds for SLC and FLC tasks, respectively. The development set of each the folds is represented by dev (internal); however, the un-annotated corpus used in leaderboard comparisons by dev (external). We remove empty and single token sentences after tokenization. Experimental Setup: We use PyTorch framework for the pre-trained BERT model (Bert-base-cased), fine-tuned for SLC task. In the multi-granularity loss, we set $\\alpha = 0.1$ for sentence classification based on dev (internal, fold1) scores. We use BIO tagging scheme of NER in FLC task. For CNN, we follow DBLP:conf/emnlp/Kim14 with filter-sizes of [2, 3, 4, 5, 6], 128 filters and 16 batch-size. We compute binary-F1and macro-F1 BIBREF12 in SLC and FLC, respectively on dev (internal).\nExperiments and Evaluation ::: Results: Sentence-Level Propaganda\nTable TABREF10 shows the scores on dev (internal and external) for SLC task. Observe that the pre-trained embeddings (FastText or BERT) outperform TF-IDF vector representation. In row r2, we apply logistic regression classifier with BERTSentEmb that leads to improved scores over FastTextSentEmb. Subsequently, we augment the sentence vector with additional features that improves F1 on dev (external), however not dev (internal). Next, we initialize CNN by FastTextWordEmb or BERTWordEmb and augment the last hidden layer (before classification) with BERTSentEmb and feature vectors, leading to gains in F1 for both the dev sets. Further, we fine-tune BERT and apply different thresholds in relaxing the decision boundary, where $\\tau \\ge 0.35$ is found optimal.\nWe choose the three different models in the ensemble: Logistic Regression, CNN and BERT on fold1 and subsequently an ensemble+ of r3, r6 and r12 from each fold1-5 (i.e., 15 models) to obtain predictions for dev (external). We investigate different ensemble schemes (r17-r19), where we observe that the relax-voting improves recall and therefore, the higher F1 (i.e., 0.673). In postprocess step, we check for repetition propaganda technique by computing cosine similarity between the current sentence and its preceding $w=10$ sentence vectors (i.e., BERTSentEmb) in the document. If the cosine-similarity is greater than $\\lambda \\in \\lbrace .99, .95\\rbrace $, then the current sentence is labeled as propaganda due to repetition. Comparing r19 and r21, we observe a gain in recall, however an overall decrease in F1 applying postprocess.\nFinally, we use the configuration of r19 on the test set. The ensemble+ of (r4, r7 r12) was analyzed after test submission. Table TABREF9 (SLC) shows that our submission is ranked at 4th position.\nExperiments and Evaluation ::: Results: Fragment-Level Propaganda\nTable TABREF11 shows the scores on dev (internal and external) for FLC task. Observe that the features (i.e., polarity, POS and NER in row II) when introduced in LSTM-CRF improves F1. We run multi-grained LSTM-CRF without BERTSentEmb (i.e., row III) and with it (i.e., row IV), where the latter improves scores on dev (internal), however not on dev (external). Finally, we perform multi-tasking with another auxiliary task of PFD. Given the scores on dev (internal and external) using different configurations (rows I-V), it is difficult to infer the optimal configuration. Thus, we choose the two best configurations (II and IV) on dev (internal) set and build an ensemble+ of predictions (discussed in section SECREF6), leading to a boost in recall and thus an improved F1 on dev (external).\nFinally, we use the ensemble+ of (II and IV) from each of the folds 1-3, i.e., $|{\\mathcal {M}}|=6$ models to obtain predictions on test. Table TABREF9 (FLC) shows that our submission is ranked at 3rd position.\nConclusion and Future Work\nOur system (Team: MIC-CIS) explores different neural architectures (CNN, BERT and LSTM-CRF) with linguistic, layout and topical features to address the tasks of fine-grained propaganda detection. We have demonstrated gains in performance due to the features, ensemble schemes, multi-tasking and multi-granularity architectures. Compared to the other participating systems, our submissions are ranked 3rd and 4th in FLC and SLC tasks, respectively.\nIn future, we would like to enrich BERT models with linguistic, layout and topical features during their fine-tuning. Further, we would also be interested in understanding and analyzing the neural network learning, i.e., extracting salient fragments (or key-phrases) in the sentence that generate propaganda, similar to pankajgupta:2018LISA in order to promote explainable AI."

    input = "Which basic neural architecture perform best by itself?"

    example = template.format(context=context, input=input)

    print(f"input text={example}")

    model_input_ids = tokenizer(example, return_tensors=None, add_special_tokens=True)["input_ids"]

    sequence = model_input_ids
    if len(model_input_ids) > 7950:
        half = int(7950 / 2)
        sequence = model_input_ids[:half] + model_input_ids[-half:]



    with torch.no_grad():
        full_input_ids = torch.tensor(sequence).unsqueeze(0).to(model.device)
        full_input_dict = {"input_ids": full_input_ids, "attention_mask": torch.ones_like(full_input_ids)}


        category = classifier(example, bert_tokenizer, bert_classifier, model.device)

        if category == "qa":
            window_select_strategy = "max"
        else:
            window_select_strategy = "average"


        for i in range(model.config.num_hidden_layers):
            if not hasattr(model.model.layers[i].self_attn.config, 'window_select_strategy'):
                model.model.layers[i].self_attn.config.window_select_strategy = window_select_strategy
                
        pred = model.generate(**full_input_dict, max_new_tokens=128, do_sample=False)[:, min(full_input_ids.shape[1], 7950):]

        pred_text = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(f"pred_text={pred_text}")
            


if __name__ == "__main__":
    main()

