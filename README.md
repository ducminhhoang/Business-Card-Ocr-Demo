# Business Card OCR Demo Application

This is a Flask-based web application that demonstrates business card OCR (Optical Character Recognition) functionality. The application allows users to upload images of business cards and automatically extracts contact information using OCR technology.

## Features

### User Features
- User authentication (register, login, logout)
- Upload business card images
- Automatically extract contact information
- Edit and verify extracted information
- Save and export contact information in VCF format
- View history of uploaded business cards

### Admin Features
- View all uploaded business cards across users
- Monitor processing status
- View processing logs
- Edit extracted information
- Export card information to CSV

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/business-card-ocr-demo.git
cd business-card-ocr-demo
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Install Tesseract OCR:

   - **Windows:**
     Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

   - **macOS:**
     ```
     brew install tesseract
     ```

   - **Linux:**
     ```
     sudo apt-get install tesseract-ocr
     ```

5. Run the application:
```
flask run
```

6. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage

### User Flow
1. Register for an account or log in
2. Upload a business card image on the dashboard
3. Review and edit the extracted information
4. Save the information
5. Export the contact to VCF format if needed

### Admin Flow
1. Log in as an admin user (the first registered user is automatically an admin)
2. Access the admin dashboard
3. View all uploaded cards across users
4. Edit information, view logs, or export data as needed

## Project Structure

```
business-card-ocr-demo/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── src/                   # Static files (CSS, JS, uploads)
│   ├── run/               # Main processing files
│   └── utils/             # Feature files
├── static/                # Static files (CSS, JS, uploads)
│   ├── uploads/           # Uploaded business card images
│   └── exports/           # Exported contact files
└── templates/             # HTML templates
    ├── base.html          # Base template with layout
    ├── index.html         # Homepage
    ├── login.html         # Login page
    ├── register.html      # Registration page
    ├── dashboard.html     # User dashboard
    ├── edit_card.html     # Edit card information page
    ├── admin_dashboard.html # Admin dashboard
    └── view_logs.html     # View processing logs page
```

## Technologies Used

- Flask - Web framework
- SQLAlchemy - ORM for database operations
- Flask-Login - User authentication
- Tesseract OCR - Optical character recognition engine
- OpenCV - Image processing
- Bootstrap - Frontend framework
- Pandas - Data manipulation and export

## Notes

- This is a demo application and may require additional refinement for production use
- The OCR accuracy depends on the quality of uploaded images and may require adjustments for different business card layouts
- The first registered user becomes an admin by default