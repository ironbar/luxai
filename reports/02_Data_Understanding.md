# Data Understanding

## Collect initial data

<!---Acquire the data (or access to the data) listed in the project resources.
This initial collection includes data loading, if necessary for data understanding.
For example, if you use a specific tool for data understanding, it makes perfect
sense to load your data into this tool. This effort possibly leads to initial data
preparation steps.
List the dataset(s) acquired, together with their locations, the methods used to
acquire them, and any problems encountered. Record problems encountered and any
resolutions achieved. This will aid with future replication of this project or
with the execution of similar future projects.

>	Indeed it's a pain downloading huge files. Especially when there are connection issues. I used "wget" to download the dataset with an option "-c" for resuming capability in case the download fails.  You would need to save the cookies in the page using a chrome extension Chrome Extension  save the cookies as cookies.txt from the extension  Then you can download the files by using the following command

	wget -c -x --load-cookies cookies.txt https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data?train_wkt.csv.zip

--->

This is a simulation competition. It is possible to download games using the Kaggle API, but the most
common way to gather data is playing the game.

It seems that the game engine is written in Javascript, and I have read in the [forum](https://www.kaggle.com/c/lux-ai-2021/discussion/267351#1486329)
that translating that to python could yield speedups.

### Engine installation

- [Getting started](https://github.com/Lux-AI-Challenge/Lux-Design-2021#getting-started)
- [Install node js](https://github.com/nodesource/distributions/blob/master/README.md#installation-instructions)
- [Python get started](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits/python)

### Useful resources

- [Sample folder](https://github.com/Lux-AI-Challenge/Lux-Design-2021/raw/master/kits/python/simple/simple.tar.gz)
- [Visualizer](https://2021vis.lux-ai.org/) allows to upload replays
- [Run tournaments with trueskill rating](https://github.com/Lux-AI-Challenge/Lux-Design-2021#cli-leaderboard-evaluation)
- [How to make a submission](https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits/python#submitting-to-kaggle). It is different from hungry geese, we need to submit a .tar.gz file with
a `main.py` script. This allows to save the models to file instead of embedding them in the code.

## External data

<!--- It is allowed in this challenge? If so write it here ideas of how to find
it and if people have already posted it on the forum describe it. --->

## Describe data

<!---Describe the data that has been acquired, including the format of the data,
the quantity of data (for example, the number of records and fields in each table),
the identities of the fields, and any other surface features which have been
discovered. Evaluate whether the data acquired satisfies the relevant requirements. --->

## Explore data

<!---This task addresses data mining questions using querying, visualization,
and reporting techniques. These include distribution of key attributes (for example,
the target attribute of a prediction task) relationships between pairs or small
numbers of attributes, results of simple aggregations, properties of significant
sub-populations, and simple statistical analyses.

Some techniques:
* Features and their importance
* Clustering
* Train/test data distribution
* Intuitions about the data
--->

## Verify data quality

<!---Examine the quality of the data, addressing questions such as: Is the data
complete (does it cover all the cases required)? Is it correct, or does it contain
errors and, if there are errors, how common are they? Are there missing values in
the data? If so, how are they represented, where do they occur, and how common are they? --->

## Amount of data

<!---
How big is the train dataset? How compared to the test set?
Is enough for DL?
--->
