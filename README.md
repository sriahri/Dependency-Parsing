# Dependency-Parsing
This repository contains the implementation of a neural-network based dependency parser with the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.
## Dependency Parser
A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between
head words, and words which modify those heads. There are multiple types of dependency parsers,
including transition-based parsers, graph-based parsers, and feature-based parsers. Your implementation
will be a transition-based parser, which incrementally builds up a parse one step at a time. At every step
it maintains a partial parse, which is represented as follows:
•A stack of words that are currently being processed.
•A buffer of words yet to be processed.
•A list of dependencies predicted by the parser.
Initially, the stack only contains ROOT, the dependencies list is empty, and the buffer contains all words
of the sentence in order. At each step, the parser applies a transition to the partial parse until its buffer
is empty and the stack size is 1. The following transitions can be applied:
•SHIFT: removes the first word from the buffer and pushes it onto the stack.
•LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of
the first item and removes the second item from the stack, adding a first word → second word
dependency to the dependency list.
•RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second
item and removes the first item from the stack, adding a second word → first word dependency to
the dependency list.

On each step, your parser will decide among the three transitions using a neural network classifier.
### (a) Go through the sequence of transitions needed for parsing the sentence “I attended lectures in the NLP class"
The dependency tree for the sentence is shown below. At each step, give
the configuration of the stack and buffer, as well as what transition was applied this step and what
new dependency was added (if any). 

#### Answer: Please take a look at the write_up file for the answers.

### (b) A sentence containing n words will be parsed in how many steps (in terms of n)? Briefly explain in 1–2 sentences why?

#### Answer: The number of operations that are required to parse a sentence of n words is 2n. This is because we need to push all the n words into the stack. This requires n SHIFT operations. We need to build the dependencies by popping out the words that are present in the stack. This requires a combination of LEFT-ARC and RIGHT-ARC operations which are n in total.

The transition mechanics the parser uses are placed in the parser_transitions.py. To run non-exhaustive basic tests you can run the PartialParse class by using the following command.
#### python parser_transitions.py part_c

Our network will predict which transition should be applied next to a partial parse. We
could use it to parse a single sentence by applying predicted transitions until the parse is complete.
However, neural networks run much more efficiently when making predictions about batches of data
at a time (i.e., predicting the next transition for any different partial parses simultaneously). We
can parse sentences in minibatches with the following algorithm.
### Algorithm: Minibatch Dependency Parsing
Input: sentences, a list of sentences to be parsed and model, our model that makes parse decisions
* Initialize partial parses as a list of PartialParses, one for each sentence in sentences.
* Initialize unfinished parses as a shallow copy of partial parses.
* while unfinished parses is not empty do
* Take the first batch size parses in unfinished parses as a minibatch.
* Use the model to predict the next transition for each partial parse in the minibatch.
* Perform a parse step on each partial parse in the minibatch with its predicted transition.
* Remove the completed (empty buffer and stack of size 1) parses from unfinished parses.
* end while

Return: The dependencies for each (now completed) parse in partial parses.


To run basic non-exhaustive tests on the mini batch algorithm using the following command.
#### python parser_transitions.py part_d

To train the neural network based on the given the state of the stack,
buffer, and dependencies, which transition should be applied next.
First, the model extracts a feature vector representing the current state. We will be using the feature
set presented in the original neural dependency parsing paper: A Fast and Accurate Dependency
Parser using Neural Networks.1 The function extracting these features has been implemented for
you in utils/parser utils.py. This feature vector consists of a list of tokens (e.g., the last
word in the stack, first word in the buffer, dependent of the second-to-last word in the stack if there
is one, etc.). They can be represented as a list of integers w = [w1,w2,...,wm] where m is the
number of features and each 0 ≤ wi < |V | is the index of a token in the vocabulary (|V | is the vocabulary size). Then our network looks up an embedding for each word and concatenates them
into a single input vector:
x = [Ew1,...,Ewm] ∈Rdm
where E ∈R|V |×d is an embedding matrix with each row Ew as the vector for a particular word w.
We then compute our prediction as:
h = ReLU(xW + b1)
l = hU + b2
ˆy = softmax(l)
where h is referred to as the hidden layer, l is referred to as the logits, ˆy is referred to as the
predictions, and ReLU(z) = max(z,0)). We will train the model to minimize cross-entropy loss:
J(θ) = CE(y, ˆy) = −
3∑
i=1
yi log ˆyi
To compute the loss for the training set, we average this J(θ) across all training examples.
We will use UAS score as our evaluation metric. UAS refers to Unlabeled Attachment Score, which
is computed as the ratio between number of correctly predicted dependencies and the number of
total dependencies despite of the relations (our model doesn’t predict this)

Finally execute the following command to train your model and compute predictions on test data from Penn Treebank (annotated with Universal Dependencies).
#### python run.py 

There are also a lot of details and conventions for dependency annotation. If you want to learn more about them, you can look at the UD website: https://universaldependencies.org/ or https://people.cs.georgetown.edu/nschneid/p/UD-for-English.pdf

There are situations where the parsers like ours might be wrong. Generally, there are four types of parsing errors.

#### 1. Prepositional Phrase Attachment Error: 
A Prepositional Phrase Attachment Error is when a prepositional
phrase is attached to the wrong head word.
#### 2. Verb Phrase Attachment Error: 
In the sentence "Leaving the store unattended, I went
outside to watch the parade", the phrase "leaving the store unattended" is a verb phrase. A Verb Phrase Attachement Error is when a verb phrase is attached to the wrong head word. In this example, the correct head word is "went".
#### 3. Modifier Attachment Error: 
In the sentence "I am extremely short", the adverb extremely is a modifier of the adjective "short". A Modifier Attachment Error is when a modifier is attached to the wrong head word. In this example, the correct head word is "short".
#### 4. Coordination Attachment Error:
A Coordination Attachment Error is when the second conjuct is attached to the wrong head word.
### References:
https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf
https://www.grammarly.com/blog/prepositional-phrase/
https://examples.yourdictionary.com/verb-phrase-examples.html
https://universaldependencies.org/docsv1/
