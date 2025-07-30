# AI-Powered Error Log Classification & Self-Learning

**LogSenseAI** is an AI/ML-powered tool built on a fine-tuned BERT model designed to intelligently analyze and classify various types of error logs. It doesn’t just stop at classification — it continually learns and adapts over time, becoming smarter with each new log.

---

## 🚀 Key Features

- 🧠 **Deep Log Understanding**: Utilizes a BERT-based model for advanced natural language understanding of logs.
- 🗂️ **Error Classification**: Automatically categorizes logs into meaningful error types.
- 🔁 **Self-Learning Engine**: Continuously retrains itself with historical logs to improve accuracy and adapt to evolving systems.
- ⚙️ **Flexible Integration**: Easily integrates with CI/CD pipelines and monitoring systems for real-time insights.

---

## 📦 Use Cases

- Monitoring system and application logs to detect emerging error patterns.
- Analyzing crash logs to expedite root cause analysis.
- Supporting automated categorization within observability and incident response tools.
- Enhancing proactive error detection to improve system reliability.

---

## 🛠️ How It Works

1. Ingests raw error logs from various sources.
2. Applies a fine-tuned BERT-based NLP model to interpret and understand log content.
3. Classifies each log entry into predefined or dynamically learned error categories.
4. Stores classifications and feedback for continual model refinement.

---

## 🔄 Continual Learning

The model continuously strengthens itself by:

- Detecting drifts and changes in log patterns over time.
- Improving accuracy using evolving, domain-specific data.
- Building resilience and adaptability for long-term deployment.
- Incorporating feedback loops for supervised retraining.

---

## 📥 Installation

1. Clone this repository:

   git clone https://github.com/yashdasari1/AI-LogAnalysis.git

   cd ./AI-LogAnalysis

3. Install dependencies:

   pip install -r requirements.txt

5. Configure your log sources and model parameters

---

## 🧑‍💻 Usage

- Run a script to initially train your model with exsting trained logs, (Run only once at first time)

  python Error_Classifier_Initial_Training.py

- Execute the classification engine: (Enter the New log path manually for now in the code and Run)

  python Analyze_Logs_Errors.py

- The model classifies the error and re-trains your model, Gets better for next one

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests to help improve "LogSenseAI".

---

## 📞 Contact

For questions or support, please open an issue on GitHub or contact yashwanthdasari10@gmail.com.

---

*Thanks for using LogSenseAI — bringing intelligence and adaptability to your log analysis!*
