# Intelligent Thesis Evaluation System

## Project Overview
An AI-powered academic paper evaluation system supporting multi-model analysis and comprehensive scoring.

## Key Features
- Intelligent PDF paper analysis
- Multi-model support (GPT-4o, GPT-3.5, Deepseek)
- Multi-role evaluation (Computer Science, Library Science, Information Management)
- Detailed scoring reports generation
- Visualization of evaluation results

## Technology Stack
- Python 3.8+
- Flask
- OpenAI API
- Deepseek API
- Matplotlib
- Pandas
- PyPDF2

## Project Structure
Structure
thesis_evaluation_en/
├── templates/
│ └── index.html # Frontend HTML template
├── utils/
│ └── progress.py # Progress tracking utility
├── standards/ # Evaluation standards
│ └── evaluation_standards.xlsx
├── app.py # Flask application entry
├── thesis_evaluation.py # Core evaluation logic
└── requirements.txt # Dependency l

## Installation Steps
1. Clone the repositoryist
bash
git clone https://github.com/yangyuwen-bri/thesis-evaluation.git
cd thesis-evaluation

2. Create virtual environment
bash
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
bash
pip install -r requirements.txt

4. Configure API Keys
- Add OpenAI and Deepseek API keys in `config.json`

5. Run the application
bash
python app.py

## Usage Guide
1. Upload PDF thesis
2. Select evaluation model and role
3. Start paper analysis
4. View detailed evaluation report

## System Requirements
- Python 3.8+
- OpenAI API Key
- Deepseek API Key

## Dependencies
See `requirements.txt` for complete list

## Evaluation Dimensions
- Research Topic
- Methodology
- Content Quality
- Research Findings
- Writing Standards

## Supported Roles
- Computer Science Professor
- Library Science Professor
- Information Management Professor
- Cybersecurity Professor
- Economics Professor
- Journalism Professor
- Custom Role Support

## Supported AI Models
- GPT-4o
- GPT-3.5-turbo
- Deepseek Chat

## License
MIT License

## Contribution Guidelines
1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## Contact
Project Maintainer: YuwenYang
Email: brilliantyangmiss@outlook.com

## Acknowledgements
Thanks to all developers and researchers contributing to this project.
