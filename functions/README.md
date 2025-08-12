# Mefron PCB Analysis Tool

This project provides a robust framework for developing an AI-powered tool for Mefron to assist in determining quality check requirements and certification needs for user-defined PCB boards based on uploaded images.

**IMPORTANT NOTE:**
This repository provides the complete infrastructure: the Streamlit user interface, data handling scripts, model training examples, and a structured approach for your dataset. However, the core functionality of analyzing PCB images to accurately determine "quality check required" or "certification needed" **heavily relies on a custom-trained Machine Learning (ML) model**. You must collect, label, and train your own PCB image dataset specific to Mefron's products and quality standards to make this tool fully functional and accurate.

## Features

*   **Modular Design:** Separate scripts for dataset creation, loading, data augmentation, and model training.
*   **Streamlit UI:** User-friendly web interface for uploading PCB images and selecting analysis options.
*   **Two ML Tasks:**
    *   **Quality Check Required:** Classifies the level of quality check needed (e.g., basic, enhanced, comprehensive).
    *   **Certification Needed:** Identifies required certifications (e.g., CE, RoHS, UL, FCC) using multi-label classification.
*   **Extensible:** Designed to easily integrate your custom-trained ML models.

## Project Structure
