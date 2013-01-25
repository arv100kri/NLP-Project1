NLP-Project1
============

The project has the following structure:
1. A directory called src which contains the code for the classiﬁers and for the plotting functionality.
2. A directory called Response which contains the response ﬁles of my “main” classiﬁer and “special” classiﬁer. My best “main” classiﬁer was the Naive Bayes, with a smoothing parameter of α = 10−3. I tried to improve on the Naive Bayes Classiﬁer according to the paper mentioned in the project speciﬁcation. My best classiﬁer from that was the Complement Naive Bayes classiﬁer, with the same smoothing parameter as above. (More details in the report)
3. A directory called generated ﬁles which contain ﬁles where I have dumped the data produced by the classiﬁers. Some of the ﬁles in that directory are used to plot the data, and some are used to just to verify the metrics.
4. Directories train, dev and test contain the training, development and test data respectively.
5. The ﬁles train.key, dev.key and scorer.py are used to obtain the accuracy metrics of the training and development data.
6. The ﬁle sentiment-vocab.tﬀ is the sentiment vocabulary
