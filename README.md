# Multi-Agent Exploratory Data Analysis (EDA) System

This project implements a sophisticated multi-agent system to streamline and automate the Exploratory Data Analysis (EDA) process. The system uses specialized AI agents to handle different aspects of data analysis, ensuring comprehensive coverage and high-quality results.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for different EDA tasks
- **AutoGen AgentChat Framework**: Official Microsoft AutoGen implementation
- **Streamlit Web Interface**: Beautiful, interactive web dashboard
- **Automated Workflow**: Streamlined process from data loading to final report
- **Professional Reports**: Comprehensive EDA reports in .docx format
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover features
- **Modular Design**: Easy to extend and customize for specific needs

## 🤖 Agent Roles

1. **DataLoader Agent**: Loads and validates datasets, provides data information
2. **DataAnalyzer Agent**: Performs comprehensive statistical analysis and insights
3. **DataVisualizer Agent**: Creates professional charts and visualizations
4. **ReportGenerator Agent**: Compiles comprehensive EDA reports in .docx format

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Required Python packages (see requirements.txt)

## 🏗️ Framework

This project uses the **official Microsoft AutoGen AgentChat framework** as documented at [https://microsoft.github.io/autogen/stable//index.html](https://microsoft.github.io/autogen/stable//index.html).

- **AgentChat**: Programming framework for building conversational multi-agent applications
- **Built on Core**: Event-driven programming framework for scalable multi-agent AI systems
- **Extensions**: OpenAI integration for model access and agent communication

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your-actual-api-key-here`
   - Or set environment variable: `export OPENAI_API_KEY=your-api-key-here`

## 📊 Usage

### Option 1: Streamlit Web App (Recommended)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**:
   ```bash
   python run_streamlit.py
   ```
   Or manually:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Use the web interface**:
   - Upload your CSV file
   - Enter your OpenAI API key
   - Click "Start EDA Analysis"
   - View results in the interactive dashboard

### Option 2: Command Line Interface

1. **Prepare your dataset**: Place your CSV file in the project directory
2. **Update the file path**: Modify the `file_path` variable in `main()` function
3. **Run the system**:
   ```bash
   python app.py
   ```

### Customization

- **Modify agent behavior**: Edit the system messages in `_create_agents()`
- **Change termination conditions**: Modify the termination logic
- **Add new agents**: Extend the agent creation method
- **Customize requirements**: Modify the `specific_requirements` in the main function

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Configuration

- **Default Model**: GPT-4o (configured in the code)
- **Customization**: Modify the `model_name` parameter in `MultiAgentEDASystem.__init__()`
- **Alternative Models**: GPT-4-turbo, GPT-3.5-turbo (update in code)
- **Framework**: Uses official Microsoft AutoGen AgentChat framework

## 📁 File Structure

```
EDA_MAS/
├── app.py                 # Main application file (CLI version)
├── streamlit_app.py       # Streamlit web application
├── run_streamlit.py       # Streamlit launcher script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
├── sample_customer_data.csv  # Your dataset file
└── eda_plots/            # Generated visualizations
```

## 🎯 Workflow Process

1. **Data Loading**: DataLoader Agent loads and validates your dataset
2. **Data Analysis**: DataAnalyzer Agent performs comprehensive statistical analysis
3. **Visualization**: DataVisualizer Agent creates professional charts and plots
4. **Report Generation**: ReportGenerator Agent compiles comprehensive .docx report
5. **Final Output**: Complete EDA report with visualizations and insights

## 📈 Expected Output

The system will generate:
- **Data quality assessment** with preprocessing details
- **Statistical summaries** for all variables
- **Visualizations** appropriate for your data types
- **Insights and patterns** discovered in the data
- **Comprehensive EDA report** in Microsoft Word (.docx) format ready for presentation
- **Visualization plots** saved as high-quality PNG files

## ⚠️ Important Notes

- **API Costs**: Running this system will consume OpenAI API credits
- **Data Privacy**: Ensure your data doesn't contain sensitive information
- **File Paths**: Use absolute paths or ensure relative paths are correct
- **Memory Usage**: Large datasets may require significant memory

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file is properly configured
2. **File Not Found**: Check the file path in the `main()` function
3. **Import Errors**: Verify all dependencies are installed
4. **Memory Issues**: Consider using smaller datasets for testing

### Getting Help

- Check the console output for detailed error messages
- Verify your OpenAI API key is valid and has sufficient credits
- Ensure your dataset file is accessible and properly formatted

## 🔮 Future Enhancements

- **Additional Agent Types**: Domain-specific analysis agents
- **Custom Visualization Templates**: Industry-specific chart styles
- **Export Options**: Multiple report formats (PDF, HTML, etc.)
- **Integration**: Connect with other data science tools
- **Performance Optimization**: Parallel processing for large datasets

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to extend the system with additional agents, visualization types, or analysis methods. The modular design makes it easy to add new functionality.

---

**Happy Data Analysis! 🎉**
