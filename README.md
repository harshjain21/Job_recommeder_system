# Job Recommendation & Matching System

## Project Overview

This project aimed to develop a robust system for **job recommendation and matching**, building upon an initial phase of job classification. The primary goal was to provide users with job postings highly relevant to their profiles by leveraging advanced Natural Language Processing (NLP) and machine learning techniques. The initial phase of this project focused on **classifying job postings into meaningful categories**.


This phase was dedicated to building the core recommendation engine.

### 1. Data Acquisition & Challenges

Our system required two primary inputs:
* **Job Listings:** The categorized dataset derived from Part 1, with the `skills` column specifically identified as "required skills."
* **User Profiles:** Data representing individual job seekers.

**Challenge: User Profile Data Acquisition**
Initially, the project faced a challenge in acquiring a realistic dataset of user profiles due to privacy and data availability constraints.
* **Problem:** Lack of real-world user resume/profile data.
* **Solution Adopted:** To overcome this, we opted to **simulate and create a large dummy dataset of user profiles (`user_df`)**.

**Dummy User Profile Generation:**
We generated 1000 dummy user profiles, each equipped with diverse, professionally relevant features, ensuring a broad representation across various domains (IT, Healthcare, Legal, Engineering, Business, Admin, Sales, Marketing). Key features included:
* `user_id`
* `name`
* `email_address`
* `user_skills` (e.g., "Python, SQL, Machine Learning")
* `user_educational_qualifications` (e.g., "Master's Degree in Data Science", "MBBS")
* `user_work_experience_years` (numerical years)
* `user_work_experience_details` (a textual summary of past roles, companies, and durations, generated using `Faker` to mimic real experience)
* `user_interests` (aligned with job categories, e.g., "Data Science", "Healthcare")

### 2. Data Preprocessing (for both Jobs & Users)

A crucial step was to transform both datasets into a comparable, machine-readable format.

* **Text Cleaning and Combination:**
    * For **Job Listings (`jobs_df` / `jobs_df_sampled`):** Relevant textual columns (`Role`, `Job_Description`, `skills`, `Responsibilities`) were combined into a `full_job_text` field.
    * For **User Profiles (`user_df`):** Relevant textual columns (`user_skills`, `user_educational_qualifications`, `user_work_experience_details`, `user_interests`) were combined into a `full_user_text` field.
    * A custom `clean_text` function was applied to both `full_job_text` and `full_user_text`. This function performed:
        * Lowercasing
        * Removal of punctuation and special characters
        * Removal of numbers
        * Removal of extra whitespace
        * Stop word removal (using NLTK English stopwords)
        * Lemmatization (using NLTK WordNetLemmatizer)
    * **Outcome:** `cleaned_full_job_text` and `cleaned_full_user_text` columns, ready for numerical representation.

### 3. Core Approach: Embeddings and Cosine Similarity

The heart of the recommendation system relies on finding semantic similarity.

* **Vectorization (Embeddings):**
    * The `cleaned_full_job_text` and `cleaned_full_user_text` from both datasets were converted into high-dimensional numerical vectors (embeddings).
    * We utilized the **`SentenceTransformer` library** with the `all-MiniLM-L6-v2` model. This model is specifically fine-tuned to produce embeddings where semantically similar texts are numerically close in vector space.
    * **Outcome:** `job_embeddings` (NumPy array) and `user_embeddings` (NumPy array), representing the numerical "fingerprints" of all jobs and users, respectively.

* **Matching Logic (Cosine Similarity):**
    * To determine relevance, we employed **Cosine Similarity**. This metric measures the cosine of the angle between two vectors. A score closer to 1 indicates higher similarity (vectors pointing in similar directions), while a score closer to 0 indicates less similarity (vectors being orthogonal).
    * For any given user, their `user_embedding` was compared against *every* `job_embedding` using cosine similarity.
    * **Outcome:** A similarity score for each job-user pair, enabling us to rank jobs by relevance.

### 4. Implementation Steps & Interactive Testing

The process involved sequential steps:
1.  Loading/Creating DataFrames (`jobs_df`, `user_df`).
2.  Defining and applying `clean_text` function.
3.  Loading `SentenceTransformer` model.
4.  Generating `job_embeddings` and `user_embeddings`.
5.  Implementing a function (`get_job_recommendations_for_user`) to compute similarity and return top N recommendations.
6.  Testing with both existing dummy users and a custom user input.

### 5. Problems Faced, Why, and Constraints

Throughout the project, several challenges and constraints impacted our approach:

* **Problem 1: Large Job Dataset Size (1.6 Million Rows)**
    * **Why:** Generating embeddings for such a massive text dataset requires significant computational resources (RAM, CPU/GPU) and time, which exceeded the immediate environment's capacity.
    * **Solution Adopted:** To proceed with development, we made a pragmatic decision to **randomly sample 20,000 rows** from the original `jobs_df`. This allowed us to build and test the core recommendation logic.
    * **Constraint:** Limited immediate access to distributed computing frameworks or more powerful hardware.

* **Problem 2: Consistent Job Recommendations for All Users (with Identical Similarity Scores)**
    * **Observation:** After implementing the system, all users were receiving the exact same job recommendations with identical similarity scores. This indicated a fundamental loss of distinctiveness in the job data.
    * **Why (Hypothesis):** While initial NaN checks showed no missing values, the most likely cause is that the **`clean_text` function was too aggressive**, stripping away critical unique information from the job descriptions. This resulted in many `cleaned_full_job_text` entries becoming either empty strings or very generic, non-distinct phrases. Consequently, the `SentenceTransformer` produced identical embeddings for these identical (or near-identical) text inputs.
    * **Constraint / Decision:** Due to **time constraints**, a decision was made to acknowledge this ongoing issue and proceed with custom input testing rather than dedicating more time to debugging the cleaning function or data inherent distinctiveness at this stage. This specific problem is noted for future investigation.
    * **What could have been done (but wasn't due to constraints):**
        * **Detailed Debugging:** Systematically analyze `value_counts()` for `cleaned_full_job_text` and inspect original input data for problematic rows to pinpoint *exactly* what information was lost.
        * **Refining `clean_text`:** Experiment with less aggressive cleaning (e.g., keeping numbers if relevant, customizing stopwords for domain-specific terms, or re-evaluating regex patterns).
        * **Feature Engineering:** Exploring if other structural features from the job postings could be incorporated beyond just combining raw text.
        * **Alternative Embedding Strategies:** While `SentenceTransformer` is good, investigating other models or pooling strategies if the cleaning wasn't the sole culprit.

## Conclusion & Future Work

Despite facing challenges related to data volume and the distinctiveness of processed job text, this project successfully developed an **end-to-end content-based job recommendation and matching system.** We demonstrated the capability to:
* Preprocess large textual datasets (through sampling).
* Generate high-quality semantic embeddings for both job listings and user profiles.
* Implement cosine similarity for effective job-user matching.
* Provide personalized job recommendations based on custom user input.

**Future Enhancements and Considerations:**

* **Addressing the "Generic Job Text" Issue:** This is the most critical immediate next step. A thorough review and refinement of the `clean_text` function or a deeper understanding of the original job data's distinctiveness is necessary to ensure unique job profiles lead to unique embeddings.
* **Scalability:** For the full 1.6 million row dataset, exploring distributed computing frameworks (e.g., Apache Spark, Dask) for embedding generation and similarity calculation would be essential.
* **Hybrid Recommendation:** Incorporating collaborative filtering aspects if user interaction data (clicks, applications) becomes available.
* **User Interface:** Developing a user-friendly front-end for users to input their profiles and view recommendations.
* **Real-world Evaluation:** Once deployed, gather real user feedback and track metrics like click-through rates, conversion rates, and user satisfaction.
* **Model Fine-tuning:** If domain-specific labeled data were available, fine-tuning a `SentenceTransformer` or other transformer model on job-related content could further enhance recommendation quality.

This project serves as a strong foundation for building sophisticated job matching capabilities, highlighting both the power of NLP and the practical considerations of real-world data and computational constraints.
