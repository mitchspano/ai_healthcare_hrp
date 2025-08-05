# T1D Chat UI - Modern Diabetes Assistant

A modern, responsive chat interface for Type 1 Diabetes management powered by AI.

## ✨ Features

### 🎨 Modern Design

- **Clean, professional interface** with a health-focused color scheme
- **Responsive design** that works on desktop and mobile devices
- **Smooth animations** and transitions for better user experience
- **Accessibility features** including keyboard navigation and screen reader support

### 💬 Chat Interface

- **Real-time messaging** with the AI diabetes assistant
- **Message timestamps** for conversation tracking
- **Loading indicators** with animated dots
- **Error handling** with user-friendly error messages
- **Auto-scroll** to latest messages

### 🚀 Quick Actions

Pre-configured buttons for common diabetes-related queries:

- 📊 Blood sugar status
- 🍽️ Dietary recommendations
- 🏃‍♂️ Exercise advice
- 💊 Medication reminders
- ⚠️ Emergency symptoms
- 📈 Weekly summaries

### 📊 Health Metrics Dashboard

- **Current blood glucose** reading with trend indicator
- **Average glucose** levels
- **Target range** display
- **Insulin on board** (IOB) tracking
- **Last reading and meal** timestamps
- **Collapsible** for space management

### 🚨 Emergency Features

- **Emergency button** for urgent situations
- **Visual indicators** for system status
- **Quick access** to critical information

## 🛠️ Technology Stack

- **React 19** - Modern React with hooks
- **Tailwind CSS** - Utility-first CSS framework
- **Vite** - Fast build tool and dev server
- **Inter Font** - Clean, readable typography

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd t1d-chat-ui
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:

   ```env
   VITE_API_URL=http://localhost:8000/chat
   ```

4. **Start the development server**

   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to `http://localhost:5173`

## 🔧 Configuration

### API Endpoint

The UI connects to your backend API. Update the `VITE_API_URL` environment variable to point to your API server.

### Subject ID

Currently hardcoded to "Subject 9". In a production app, this would come from user authentication.

## 📱 Usage

### Basic Chat

1. Type your message in the input field
2. Press Enter or click the send button
3. View the AI assistant's response

### Quick Actions

1. Click any quick action button in the sidebar
2. The message is automatically sent to the assistant
3. Receive instant, relevant responses

### Health Metrics

1. View current health status in the sidebar
2. Toggle metrics visibility with the "Hide/Show Metrics" button
3. Monitor key indicators at a glance

### Emergency

1. Click the red "Emergency Help" button for urgent situations
2. Follow the prompts for immediate assistance

## 🎨 Customization

### Colors

The app uses a health-focused color palette:

- **Primary Blue**: `#3b82f6` (Trust, medical)
- **Success Green**: `#10b981` (Health, positive)
- **Warning Red**: `#ef4444` (Emergency, alerts)
- **Neutral Grays**: For text and backgrounds

### Styling

All styles are built with Tailwind CSS classes. You can customize:

- Colors in `tailwind.config.js`
- Typography in `src/index.css`
- Component styles in `src/App.css`

## 🔒 Security Considerations

- **Input validation** on all user inputs
- **Error handling** for API failures
- **No sensitive data** stored in localStorage
- **HTTPS recommended** for production

## 🧪 Testing

```bash
# Run linting
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📈 Performance

- **Lazy loading** of components
- **Optimized images** and assets
- **Minimal bundle size** with Vite
- **Fast refresh** during development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support or questions:

- Check the documentation
- Open an issue on GitHub
- Contact the development team

---

**Note**: This is a healthcare application. Always consult with medical professionals for medical advice. This tool is designed to assist, not replace, professional medical care.
