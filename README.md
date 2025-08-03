

##  **Data Science & Artificial Intelligence Portfolio**



Welcome to my portfolio!

My name is Alexi Kehayias,  a third year Data Science & A.I student at Breda University of Applied Sciences

My background is made up of mathematics, computer science and economics from Cambridge Advanced and Ordinary Levels, alongside technical experience with maintaining linux servers.

I specialise in the following areas:
1. Data Analytics.
2. Computer Vision.
3. Machine Learning within the CRISP-DP Cycle.
4. Natural Language Processing (NLP).
5. Scientific Research using qualitative and quantitative methods.
6. Machine Learning Operations (MLOps) & Deployment.

---

## **1. Data Analytics: Microsoft Power BI**

### **Tuberculosis Analysis in Zimbabwe using Microsoft Power BI**

Designed an interactive and data-rich Power BI dashboard to support public health decision-making by analyzing tuberculosis (TB) trends in Zimbabwe. 

The dashboard explores key metrics such as infection rates, gender and age distribution, and provincial case breakdowns, while also benchmarking Zimbabwe’s performance against neighboring countries in Southern Africa. This comparative analysis provides regional context for assessing the country’s progress in fighting TB and reveals cross-border public health dynamics.

The dashboard is aligned with Sustainable Development Goal (SDG) 3 – Good Health and Well-Being, specifically targeting the goal of ending the TB epidemic. It tracks Zimbabwe’s progress over multiple years, allowing users to measure improvement, detect setbacks, and identify underperforming areas requiring focused intervention. Additional features include customizable filters, time series visualizations, and integrated WHO indicators to ensure accuracy and global relevance.

This project demonstrates strong skills in health data analysis, regional benchmarking, and data storytelling—enabling stakeholders to derive actionable insights, allocate resources effectively, and enhance transparency in reporting health outcomes.



**AI/Data Science Concepts**

1. Exploratory Data Analysis (EDA)
2. Data Cleaning & Wrangling
3. Data Visualization
4. Time Series Analysis
5. Public Health Analytics
6. SDG Goal Tracking
7. Regional Benchmarking


### **IKEA Dashboard For Card Skimming Detection**

Developed a comprehensive and security-focused Power BI dashboard to analyze and monitor the redeployment of payment terminals across IKEA Breda. 

This solution enables detailed tracking of terminal movements by both physical location and employee ID to uncover usage patterns, identify anomalies, and flag high-frequency redeployment zones that may signal operational inefficiencies or heightened skimming risk.

The dashboard incorporates key performance indicators such as total redeployments, frequency trends, employee-level deployment behavior, and compliance with IKEA’s mandatory twice-weekly terminal check protocol. 

Advanced visualizations help pinpoint "hotspot" areas within the store where terminals are frequently moved, facilitating deeper investigation and targeted action.

The tool not only enhances transparency and operational control but also supports proactive loss prevention, security compliance, and staff accountability. Custom filters and time-series visuals allow IKEA management to drill down into specific periods or departments, improving data-driven decision-making and aligning daily operations with broader store security strategies.

This project demonstrates advanced data storytelling, technical rigor, and a deep understanding of real-world retail challenges—particularly at the intersection of security, compliance, and customer transaction integrity.


**AI/Data Science Concepts**
1. Exploratory Data Analysis (EDA).
2. Data Cleaning.
3. Data Visualization.
4. Compliance Monitoring.
5. Security Risk Analytics.
6. Operational Intelligence.
7. Employee Behavior Analysis.

---

## **2. Computer Vision**

### ** Image-Based Classroom Occupancy Detection (with Explainable AI)**

Developed image classification appliccation to detect classroom occupancy within universities. The core model is based on a Convolutional Neural Network (CNN) trained to classify whether classrooms are occupied or unoccupied using live camera feed imagery.

The solution is designed to reduce booking conflicts, optimise space usage, and enhance the user experience for students and staff.

To support transparency and ethical deployment, the model incorporates Explainable AI (XAI) techniques such as Grad-CAM to visualize which parts of the image influenced model predictions. These visualizations confirm the model’s focus on relevant regions (e.g., seats, human figures), enabling fair and trustable outputs.

The system was designed to integrate with BUas’s booking tools, provide real-time availability insights, generate usage analytics dashboards for facilities teams, and support sustainability efforts through better space utilization. A strong emphasis was placed on privacy-first design, ensuring anonymized data handling throughout the pipeline.

This project combined technical execution with real-world value, aligning AI implementation with stakeholder needs, sustainability goals, and responsible AI principles.

**Programming Languages/Tools**
1. Python.
2. OpenCV.
3. TensorFlow / Keras.
4. Matplotlib, Seaborn.
5. Grad-CAM (Explainable AI).

**AI/Data Science Concepts**
1. Convolutional Neural Networks (CNNs)
2. Image Classification
3. Grad-CAM for Explainability
4. Data Preprocessing & Augmentation
5. Model Evaluation (Accuracy, Precision, Recall, Confusion Matrix)
6. Responsible AI (Fairness, Privacy)


---

## **Primary Root Detection and Measurement Pipeline (NPEC)**

- Engineered a full image-processing and AI pipeline to detect, segment, and measure primary root structures from plant imagery within petri dishes—supporting agricultural phenotyping and robotics-based inoculation at NPEC (Netherlands Plant Eco-phenotyping Centre).

- The pipeline performs automated image segmentation using a deep learning model (TensorFlow/Keras), followed by post-processing steps including morphological filtering, skeletonization, and component labeling to extract precise measurements of individual roots.
- Roots are classified, measured, and linked to specific plants based on their spatial bin within the petri dish.

- It also included additional functionality such as accurate localization of root tips and origins using Euclidean skeleton path tracing.

- Filtering for roots that are moderately vertical and of sufficient length, improving biological relevance.
- Generation of skeleton overlays and annotated output images.
- Projection of root tip coordinates into robotic arm coordinate space, enabling automated lab-based tasks (e.g., root inoculation using a reinforcement learning-controlled robotic agent).

- Export of root metadata (length, position, root ID) into structured formats for further research analysis.

- The pipeline is optimized for batch processing, integrates Explainable AI techniques like overlay visualization, and aligns with smart agriculture initiatives by bridging AI-driven image analysis with physical automation in lab settings.

**Programming Languages/Tools**
1. Python
2. OpenCV
3. TensorFlow / Keras
4. NumPy / SciPy / Skimage
5. Stable-Baselines3 (PPO for Reinforcement Learning)
6. Custom Robotics Environment (OT2Env)

**AI/Data Science Concepts**
1. Image Segmentation & Morphological Processing
2. Skeletonization & Graph-Based Path Analysis
3. Object Detection & Component Labeling
4. Feature Extraction (Length, Tip/Start Localization)
5. Coordinate Transformation (Image to Robot Space)
6. Transfer Learning
7. Automated Phenotyping in Agriculture
8. Integration of AI & Robotics in Lab Automation

---

##  **3. Machine Learning within the CRISP-DP Cycle**

### **Accident Detection & Analysis Using the CRISP-DM Cycle**

- Designed and deployed a full-stack driving risk prediction system using the CRISP-DM methodology, spanning all phases from business understanding to model deployment.
- The solution is centered on classifying driving scenarios into four accident risk levels Low, Minor, Moderate, and Sever using a deep learning classifier built in Keras/TensorFlow.
- The pipeline began with collecting and cleaning raw driving behavior data from a data lake, then selectively joining only relevant tables (e.g., GPS logs, speed metrics, accident histories) using SQL queries to form a refined dataset.
- This cleaned and structured data was stored in a data warehouse, enabling consistent access for model training, monitoring, and future analytics.
- SQL logic was also used for filtering out noise, aggregating event counts, and segmenting historical behavior by location and time.

- Multiple models were explored—starting with baseline classifiers like Logistic Regression and Decision Trees, then evolving into a custom deep neural network architecture with dropout layers, batch normalization, and L2 regularization to prevent overfitting.
- After training, the best model was deployed in an interactive Streamlit web app, allowing users to simulate predictions and view real-time risk classifications based on GPS location, speed, and past accident counts.
- The app includes login functionality, dynamic UI components, risk level visualizations, and debugging panels for transparency.
- Data is scaled and processed on-the-fly before being passed to the model, with predictions shown alongside raw and normalized input values.
- This project demonstrates end-to-end capability across data engineering, modeling, and deployment—blending cloud-scale data management with real-time AI.

**Programming Languages/Tools**
1.  Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)
2. SQL (Data Lake → Warehouse ETL)
3. Streamlit (Web App & Deployment)
4. Seaborn / Matplotlib

**AI/Data Science Concepts**
1. CRISP-DM Lifecycle (Business Understanding → Deployment)
2. Data Lake to Data Warehouse Transformation
3. SQL Joins, Filtering, and Feature Aggregation
4. Deep Learning for Multi-Class Classification
5. Feature Engineering & Scaling
6. Model Regularization (Dropout, L2, BatchNorm)
7. Real-Time Prediction & Visualization
8. Streamlit-Based Interactive Dashboards
9. AI Deployment for Traffic Safety & Mobility Intelligence

---

## **4. Natural Language Processing (NLP)**

### **Emotion Detection in Greek Sentences with Translation**

Built a robust end-to-end NLP pipeline to extract, translate, and classify emotions from Greek-spoken YouTube videos—combining cutting-edge speech recognition, custom transformer models, and multilingual translation for emotion-aware content analysis.

The system takes a YouTube video as input, converts the audio to text using Whisper (large-v3) for transcription, translates the Greek transcript into English using a fine-tuned HuggingFace Greek-to-English transformer, and then predicts the emotion of each sentence using a custom bilingual emotion classification model. Each sentence is labeled with one of 7 emotions (anger, disgust, fear, happiness, neutral, sadness, surprise) along with an intensity score.

The pipeline is modular, scalable, and GPU-ready. It includes a custom OpenAI-powered scoring mechanism and was engineered to run efficiently on CUDA-enabled environments. The results are stored in a clean CSV output with aligned timestamps, raw and translated text, predicted emotion, and emotion intensity—enabling downstream analytics, content moderation, or sentiment-aware media indexing.

**Key Capabilities**
1. YouTube video → transcribed, translated, and emotion-tagged sentence-level output
2. Handles real-world noisy audio with Whisper
3. Works on Greek audio using multilingual models
4. Supports emotion intensity scoring
5. Compatible with OpenAI APIs and fine-tuned HuggingFace models
6. Output saved as structured .csv for reporting or analytics

**Programming Languages/Tools**
1. Python
2. HuggingFace Transformers
3. OpenAI API (for scoring)
4. Whisper (via whisper + FFmpeg)
5. Pandas, tqdm, NumPy

**AI/Data Science Concepts**
1. Speech-to-Text (Whisper large-v3)
2. Multilingual Machine Translation (Greek → English)
3. Transformer-Based Emotion Classification
4. Emotion Intensity Scoring
5. Text Tokenization & Preprocessing
5. Pipeline Engineering & GPU Acceleration
6. Real-Time Content Understanding

---

## **5. Scientific Research using qualitative and quantitative methods**

### **Impact of Chatbots on Small and Medium Enterprises (SMEs)**

Conducted a full-scale mixed-methods research study to explore how chatbots influence customer satisfaction and trust in Small and Medium Enterprises (SMEs), using both quantitative and qualitative methodologies. This policy paper assessed chatbot effectiveness across four key service dimensions: personalization, accuracy, responsiveness, and functionality.

The quantitative component involved hypothesis-driven survey analysis, using techniques such as Cochran’s Q test, McNemar’s post-hoc test, Levene’s test, and effect size metrics (Cohen’s d and h). Data was cleaned, tested for normality, and analyzed using both descriptive and inferential statistics to determine how personalization levels and response times impacted perceived satisfaction.

The qualitative component employed semi-structured interviews and thematic analysis (Braun & Clarke, 2006) to uncover nuanced insights about user expectations, chatbot limitations, and the trust dynamics between human vs. chatbot interactions. Themes included “Accuracy as the Core of Trust,” “Speed vs. Personalization,” and “Expectations vs. Reality.”

The study also performed a stakeholder analysis to guide SMEs on chatbot implementation strategies tailored to customer needs, resource constraints, and communication challenges.

**Key Deliverables**
1. Statistical evidence linking chatbot personalization to customer satisfaction.
2. Practical recommendations for SMEs on balancing speed, personalization, and accuracy.
3. Actionable policy suggestions to improve chatbot design, escalation paths, and trust-building mechanisms.
4. A stakeholder matrix identifying the impact of chatbot adoption across SME user groups (e.g., customers, employees, managers).

**Programming Languages/Tools**
1. Python (for survey data analysis and statistical testing)
2. Excel / Google Sheets (for data cleaning and reporting)
3. Survey Platforms (Qualatrics)
4. AI/Data Science Concepts & Research Methods:
5. Mixed-Methods Research (Quantitative + Qualitative)
6. Survey Data Cleaning & Normality Testing
7. Inferential Statistics (Cochran’s Q, McNemar’s Test, Levene’s Test, Effect Sizes)
8. Thematic Analysis (Braun & Clarke Method)
9. Business Intelligence (BI) for SMEs
10. Human-AI Interaction & Trust
11. Stakeholder Mapping & Policy Recommendations

---

## **6. Machine Learning Operations (MLOps) & Deployment**

### **Deployment of Deep Learning Pipelines via Docker, Azure & Airflow**

Developed and deployed a robust deep learning solution for root segmentation and phenotyping across multiple platforms using **end-to-end MLOps workflows**. The project—titled **ROALT (Root Analysis Toolkit)**—was built to bring innovation to plant research through scalable, reproducible, and accessible machine learning infrastructure.


**Project Features**

- Deployed a **computer vision pipeline** for root segmentation using **FastAPI** and **Gradio**, enabling both REST API access and intuitive UI for end users.
- Utilized **Docker & Portainer** for containerization, version control, and modular deployment—supporting both GPU-accelerated local inference and cloud compatibility.
- Managed automated deployment workflows via **Apache Airflow**, including scheduled tasks for data ingestion, model retraining, and batch inference.
- Integrated with **Azure ML Studio Pipelines** for cloud-based experiment tracking, dataset management, and model versioning.
- Supported both **batch and single-image inference**, with example notebooks, visualizations, and live demos.
- Enabled full CI/CD integration via **GitHub Actions**, with automated testing, Docker publishing, and linting (PEP8).

**Programming Languages / Tools**

- **Python 3.12+**
- **FastAPI** & **Gradio**
- **Docker**, **Docker Compose**, **Portainer**
- **Apache Airflow**
- **Azure ML SDK / ML Studio**
- **Git**, **GitHub Actions**
- **Poetry** (environment and dependency management)

**AI / Data Science Concepts**

1. Deep Learning for Computer Vision (CNN-based Segmentation)
2. Reproducible ML Pipelines (Development → Deployment → Monitoring)
3. RESTful ML APIs for inference and service integration
4. MLOps Lifecycle (Model Versioning, CI/CD, Automation)
5. Web UI interfaces for non-technical users (Gradio)
6. Workflow Orchestration and Automation (Airflow DAGs)

**Deployment Targets & Interfaces**

| Interface            | Platform          | Purpose                            |
|----------------------|-------------------|------------------------------------|
| **Gradio Web UI**     | Localhost (Docker) | Front-end interface for users      |
| **FastAPI Backend**   | Docker & Azure     | REST API for inference             |
| **Azure Pipelines**   | Azure ML Studio    | Cloud experiments & model tracking |
| **Airflow DAGs**      | Local (Docker)     | Pipeline automation (scheduling)   |
| **Portainer**         | Local              | Visual container management        |


This project exemplifies the application of **MLOps best practices** for real-world deep learning deployment and automation, making it suitable for research environments and production-ready systems alike.

---

