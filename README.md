
## Just-in-time Identification for Cross Project Correlated Issues

### Here we describe each file or folder in the repository.
- [**dataset**](https://github.com/hren-ron/dependency_analysis/tree/main/datasets) The datasets constructed in our study are presented in this folder, which mainly includes these 11 issue metrics we calculated on each issue in the 16 open source projects. These data can be used to build the CPC issue prediction model. Developers can use the provided code to obtain issue and commit information in the project.
- [**source_code**](https://github.com/hren-ron/dependency_analysis/tree/main/source_code) The source code used to build the model in this study is provided in this folder, which mainly includes the following functions:
    - Obtain the source data in the project from GitHub, mainly including the issue information of the project, the comment information of each issue, and the commit information in the project.
    - Obtain the Cross Project Correlated Issues (CPC) issues in the project.
    - Extract 11 different issue metrics proposed in this paper.
    - Extract text features based on TF-IDF and Word Embedding.
    - Build the CPC issue identification model based on these metrics.
    - Build mixture models based on different combinations of features.
    - Calculate the experimental results of these prediction models.
    - Calculate four types of project measures. Since the three project metrics (i.e., Basic Metrics, Dependency Metrics, and Code Complexity Metrics) can be calculated by analyzing the source code using the static code analysis tool Understand. Therefore, code for calculating another type of project metric (i.e., Ecosystem-level Metrics) is provided in this repository.
- [**manual**](https://github.com/hren-ron/dependency_analysis/tree/main/manual) This folder presents the results of our manual analysis of cross-project links. Among them, each file includes the issues that contain cross-project links and the corresponding associated reason labels.
    - The file (manual_analysis.txt) is used to show the final correlated causes of cross-project links on each issue after manual analysis.
    - The file (manual_analysis_reviewer_1.txt) is the Reviewer 1's analysis result of the causes of cross-project links on these issues.
    - The file (manual_analysis_reviewer_2.txt) is the Reviewer 2's analysis result of the causes of cross-project links on these issues.

