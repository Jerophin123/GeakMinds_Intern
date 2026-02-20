# Multi-Function AI Text Assistant

A powerful, desktop-based Generative AI text processing application that uses Machine Learning to automatically detect and process various text tasks. Built with Python, scikit-learn, and an embedded web browser interface.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This application is an intelligent text assistant that combines **Machine Learning classification** with **Generative AI text processing** capabilities. It automatically detects user intent and performs various text transformations including summarization, email formatting, paraphrasing, and title generation.

### Key Highlights

- **🤖 ML-Powered Intent Detection**: Uses Random Forest Classifier to automatically detect what task the user wants to perform
- **💬 Modern Chat Interface**: Beautiful, responsive web-based chat UI embedded in a native desktop window
- **🚀 All-in-One Application**: Single Python file containing everything - no external dependencies or separate files needed
- **📱 Portrait-Oriented Design**: Optimized for modern mobile-like interface with 9:16 aspect ratio
- **⚡ Real-Time Processing**: Instant text processing with confidence scores
- **📋 Copy-to-Clipboard**: Easy copying of results with one-click functionality

## ✨ Features

### Core Text Processing Capabilities

1. **📝 Text Summarization**
   - Automatically extracts key points from long text
   - Generates concise summaries
   - Preserves important information

2. **📧 Professional Email Formatting**
   - Converts casual text into professional email format
   - Adds proper email structure and formatting
   - Ready-to-send professional communications

3. **🔄 Text Paraphrasing**
   - Advanced synonym replacement
   - Sentence restructuring
   - Maintains original meaning while varying expression
   - Multiple paraphrasing strategies

4. **🏷️ Title Generation**
   - Creates concise, descriptive titles
   - Extracts key concepts from content
   - Optimized for readability

### User Interface Features

- **🎨 Modern Chat App Design**: Clean, white interface with blue accents
- **📱 Responsive Layout**: Adapts to different window sizes
- **⚡ Quick Action Buttons**: One-click access to common tasks
- **💡 Auto-Fill Prefixes**: Smart text input with highlighted prefixes
- **📊 Confidence Indicators**: Visual feedback on ML prediction confidence
- **📋 Copy Button**: Hover-to-reveal copy buttons on all messages
- **🔄 Auto-Detection**: Intelligent task detection without manual selection
- **📜 Scrollable History**: Full conversation history with hidden scrollbars

## 🛠️ Technology Stack

### Machine Learning
- **scikit-learn**: Random Forest Classifier for intent detection
- **TF-IDF Vectorization**: Text feature extraction
- **Pipeline Architecture**: Streamlined ML workflow

### Frontend
- **HTML5/CSS3**: Modern, responsive web interface
- **JavaScript (ES6+)**: Dynamic UI interactions
- **SVG Icons**: Scalable vector graphics for UI elements

### Backend
- **Python 3.7+**: Core application logic
- **HTTP Server**: Built-in Python HTTP server for serving the interface
- **pywebview**: Embedded native browser window
- **Threading**: Background server processing

### Data Processing
- **pandas**: Data manipulation and dataset generation
- **joblib**: Model serialization and persistence
- **Regular Expressions**: Advanced text pattern matching

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd ml-service

# Or simply download the ml-service.py file
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas>=1.5.0 scikit-learn>=1.0.0 joblib>=1.0.0 pywebview>=4.0.0
```

### Step 3: Run the Application

```bash
python ml-service.py
```

The application will:
1. Load or train the ML model (first run may take a moment)
2. Start the HTTP server on `http://127.0.0.1:8000`
3. Automatically open the embedded browser window

## 🚀 Usage

### Starting the Application

Simply run:

```bash
python ml-service.py
```

The window will open automatically. No additional configuration needed!

### Using the Chat Interface

1. **Quick Actions**: Click one of the quick action buttons (Summarize, Email, Paraphrase, Title) to auto-fill the prefix
2. **Type Your Text**: Enter the text you want to process after the prefix
3. **Send**: Click the send button or press Enter
4. **View Results**: The AI will process your text and display the result with confidence score
5. **Copy Results**: Hover over any message and click the copy button to copy to clipboard

### Example Interactions

**Summarization:**
```
User: Summarize this paragraph: [Your long text here]
Bot: [Concise summary with key points]
```

**Email Formatting:**
```
User: Rewrite this as a professional email: [Your casual message]
Bot: [Formatted professional email]
```

**Paraphrasing:**
```
User: Paraphrase this sentence: [Your text]
Bot: [Rephrased version with synonyms and restructuring]
```

**Title Generation:**
```
User: Generate a title for: [Your content]
Bot: [Concise, descriptive title]
```

### Auto-Detection Mode

The application can automatically detect your intent without using prefixes:

```
User: Can you make this shorter?
Bot: [Detects summarization task and processes accordingly]
```

## 🏗️ Architecture

### Application Structure

```
ml-service.py
├── Text Cleaning & Preprocessing
├── Dataset Generation (Synthetic Training Data)
├── ML Model Training (Random Forest Classifier)
├── Text Processing Functions
│   ├── Summarization
│   ├── Email Formatting
│   ├── Paraphrasing
│   └── Title Generation
├── Task Routing & Classification
├── HTTP Server (Background Thread)
└── Embedded Browser Window (pywebview)
```

### ML Model Architecture

1. **Feature Extraction**: TF-IDF vectorization with n-gram range (1,2)
2. **Classifier**: Random Forest with multiple decision trees
3. **Training**: Synthetic dataset with 250 samples per class
4. **Prediction**: Probability-based classification with confidence threshold (0.60)

### Data Flow

```
User Input → Text Cleaning → ML Classification → Task Routing → 
Text Processing → Result Formatting → UI Display
```

## 🤖 Machine Learning Details

### Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF vectors (1-grams and 2-grams)
- **Classes**: 4 (summarization, email, paraphrasing, title)
- **Training Data**: 1000 samples (250 per class)
- **Test Split**: 20% of data
- **Persistence**: Model saved as `rf_task_classifier.pkl`

### Model Training

The model is automatically trained on first run if no saved model exists. Training includes:
- Synthetic dataset generation
- Train-test split with stratification
- Model evaluation with accuracy and classification reports
- Automatic model persistence

### Prediction Confidence

- **High Confidence (≥60%)**: Task is executed automatically
- **Low Confidence (<60%)**: User is prompted to clarify intent
- **Confidence Score**: Displayed in the UI for transparency

## 📁 Project Structure

```
ml-service/
├── ml-service.py          # Main application file (all-in-one)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── rf_task_classifier.pkl # Saved ML model (generated on first run)
```

## 🎨 UI/UX Features

### Design Principles

- **Minimalist**: Clean, uncluttered interface
- **Modern**: Contemporary chat app aesthetics
- **Responsive**: Adapts to window resizing
- **Accessible**: Clear visual feedback and intuitive controls

### Responsive Breakpoints

- **Mobile Portrait**: Optimized for 270px+ width
- **Tablet Portrait**: Enhanced layout for 405px+ width
- **Desktop**: Full feature set for larger screens

### Visual Elements

- **Color Scheme**: White background with blue accents (#007bff)
- **Typography**: System fonts for native feel
- **Icons**: SVG icons for scalability
- **Animations**: Smooth fade-in effects for messages

## 🔧 Configuration

### Window Settings

Default window size can be modified in the main execution block:

```python
webview.create_window(
    title="Multi-Function AI Text Assistant",
    url=chatbot_url,
    width=405,              # Modify width
    height=720,             # Modify height
    resizable=True,
    min_size=(270, 480)     # Modify minimum size
)
```

### Server Port

Change the server port in the `start_server()` call:

```python
start_server(8000)  # Change port number
```

### ML Confidence Threshold

Adjust the confidence threshold in the `route_task()` function:

```python
threshold=0.60  # Change threshold (0.0 to 1.0)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `pywebview is not installed`
- **Solution**: Run `pip install pywebview`

**Issue**: Model training takes too long
- **Solution**: Reduce `samples_per_class` in `generate_dataset()` function

**Issue**: Window doesn't open
- **Solution**: Check if port 8000 is available, try a different port

**Issue**: Low prediction accuracy
- **Solution**: Retrain model by deleting `rf_task_classifier.pkl` and restarting

### Error Messages

- **"Model not loaded"**: Model file is missing or corrupted
- **"Empty input provided"**: No text was entered
- **"Uncertain request"**: ML model confidence is below threshold

## 🔒 Security Notes

- Application runs on `127.0.0.1` (localhost only)
- No external network connections required
- All processing happens locally
- No data is sent to external servers

## 📊 Performance

- **Model Loading**: < 1 second (if model exists)
- **Model Training**: ~10-30 seconds (first run only)
- **Text Processing**: < 100ms per request
- **Memory Usage**: ~50-100 MB

## 🚧 Future Enhancements

Potential improvements for future versions:

- [ ] Support for additional languages
- [ ] Custom model training with user data
- [ ] Export conversation history
- [ ] Multiple theme options
- [ ] Batch processing capabilities
- [ ] API mode for integration with other applications
- [ ] Advanced paraphrasing with multiple style options
- [ ] Integration with external LLM APIs for enhanced processing

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue in the repository.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- Uses [pywebview](https://pywebview.flowrl.com/) for embedded browser functionality
- Inspired by modern chat application interfaces

---

**Made with ❤️ for the Generative AI community**





