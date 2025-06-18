# 🤖 HR Agent - AI-Powered HR Assistant

An intelligent HR assistant powered by AI that can answer employee questions, provide policy information, and assist with HR-related inquiries using natural language processing.

## 🎥 Demo Video

Watch the HR Agent in action:

<video width="800" height="600" controls>
  <source src="HR_Agent.MOV" type="video/quicktime">
  Your browser does not support the video tag.
</video>

## 🚀 Features

### 🤖 AI-Powered Responses
- **Natural Language Processing**: Understands and responds to HR questions in natural language
- **Policy Knowledge**: Access to comprehensive HR policies and procedures
- **Context Awareness**: Maintains conversation context for better responses
- **Multi-language Support**: Can handle queries in multiple languages

### 📊 Employee Management
- **Employee Data**: Sample employee database with CSV integration
- **Query Processing**: Handles various types of HR inquiries
- **Policy Retrieval**: Quick access to HR policies and guidelines
- **Embedding Storage**: Efficient storage and retrieval of policy embeddings

### 🎨 User Interface
- **Web-based Frontend**: Clean and intuitive user interface
- **Real-time Responses**: Instant AI responses to queries
- **Conversation History**: Track previous interactions
- **Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
HR_Agent/
├── hr_agent.py           # Main HR agent application
├── agent_backend.py      # Backend logic and AI processing
├── frontend.py           # Web interface and UI components
├── store_embeddings.py   # Embedding storage and retrieval
├── requirements.txt      # Python dependencies
├── hr_policy.txt         # HR policies and procedures
├── sample_employee.csv   # Sample employee database
├── HR_Agent.MOV          # Demo video showing the application in action
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/ibraahimycl/HR_Agent.git
cd HR_Agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (if needed)
# Create a .env file with your API keys
```

## 🎮 Usage

### Running the Application
```bash
# Start the HR Agent
python hr_agent.py
```

### Web Interface
1. Open your browser and navigate to the provided URL
2. Type your HR-related questions in the chat interface
3. Receive instant AI-powered responses

### Example Queries
- "What is the company's vacation policy?"
- "How do I request time off?"
- "What are the working hours?"
- "How do I report an issue to HR?"
- "What is the dress code policy?"

## 🔧 Technical Details

### AI Components
- **Language Model**: Advanced NLP for understanding queries
- **Embedding System**: Efficient policy document search
- **Response Generation**: Context-aware answer generation
- **Memory Management**: Conversation history tracking

### Data Management
- **CSV Integration**: Employee data from CSV files
- **Policy Storage**: Text-based policy documents
- **Embedding Database**: Vector storage for quick retrieval
- **Session Management**: User session handling

### Security Features
- **Input Validation**: Secure query processing
- **Data Protection**: Employee information security
- **API Security**: Protected backend endpoints

## 📊 Sample Data

The project includes:
- **Sample Employee Database**: CSV file with employee information
- **HR Policies**: Comprehensive policy document
- **Embedding Storage**: Pre-processed policy embeddings

## 🔮 Future Enhancements

- [ ] Multi-language support expansion
- [ ] Advanced analytics dashboard
- [ ] Integration with HRIS systems
- [ ] Voice interface support
- [ ] Mobile app development
- [ ] Advanced reporting features
- [ ] Employee self-service portal

## 🛠️ Dependencies

- **OpenAI**: AI language model integration
- **Flask**: Web framework for backend
- **Pandas**: Data manipulation and CSV processing
- **NumPy**: Numerical computing
- **Requests**: HTTP library for API calls
- **Streamlit**: Web interface framework

## 📝 License

This project is developed for educational and demonstration purposes.

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Enhancing the AI capabilities

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

## 🎯 Use Cases

### HR Departments
- **Policy Queries**: Quick access to company policies
- **Employee Support**: 24/7 HR assistance
- **Onboarding**: New employee guidance
- **Compliance**: Policy compliance information

### Employees
- **Self-Service**: Quick answers to common questions
- **Policy Information**: Easy access to company policies
- **Support**: Immediate HR support
- **Guidance**: Work-related guidance and advice 