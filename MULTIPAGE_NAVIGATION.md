# Streamlit Multipage Navigation

## Overview

This application uses Streamlit's built-in multipage app feature to provide navigation between the main portioning model page and the settings page.

## Structure

```
.
├── app.py                 # Main application page (Portioning Model)
└── pages/
    └── settings.py        # Settings configuration page
```

## How It Works

Streamlit automatically creates multipage navigation when:
1. You have a main `app.py` file in the root directory
2. You have a `pages/` directory with Python files

The navigation appears automatically in the sidebar with:
- **app.py** → "Portioning Model" (main page)
- **pages/settings.py** → "Settings" (settings page)

## Navigation Features

### Automatic Sidebar Navigation
- Streamlit automatically adds page links to the sidebar
- Users can click between pages without losing application state
- Page names are derived from the file names (e.g., `settings.py` → "Settings")

### State Preservation
- Streamlit's session state persists across page navigation
- Configuration changes made in the settings page are saved to `settings.json`
- The main application loads configuration from `settings.json` on startup

### Configuration Changes
- When users save settings on the settings page, changes are written to `settings.json`
- To apply configuration changes to the main application:
  1. Save changes on the settings page
  2. Navigate back to the main page
  3. Restart the application (if needed for certain parameters)

## Running the Application

To start the application with multipage navigation:

```bash
streamlit run app.py
```

This will:
1. Start the Streamlit server
2. Open the main page (app.py) in your browser
3. Show the settings page link in the sidebar navigation

## Requirements Met

This implementation satisfies the following requirements:

- **Requirement 7.1**: Navigation mechanism provided via Streamlit's automatic sidebar
- **Requirement 7.2**: Settings page appears as a separate page in sidebar navigation
- **Requirement 7.3**: Application state is preserved when navigating to settings page
- **Requirement 7.4**: Saved configuration changes are applied when navigating away from settings page

## Notes

- No code changes are needed in `app.py` to support multipage navigation
- Streamlit handles all routing and navigation automatically
- The `pages/` directory can contain multiple pages if needed in the future
- Page order in the sidebar is determined by file names (alphabetically)
