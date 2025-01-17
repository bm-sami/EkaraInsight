# Streamlit Application for Scenario and Status Analysis

## Overview
This Streamlit application provides interactive tools for analyzing scenario and status data. Scenarios represent specific sequences of actions executed by automated robots within an application, such as logging in or accessing a homepage. Each scenario generates execution data, including:
- **Duration**: The time taken to complete each step of the scenario.
- **Status**: Whether the scenario was successfully completed (`1` for available, `0` for unavailable).

By monitoring these metrics, clients can gain valuable insights into their application's performance and availability, helping them address issues proactively. The application is designed to visualize temporal performance trends, explore execution data, and display predictive insights. It is organized into multiple pages for ease of navigation and user interaction.

## Features
### 1. **Home Page**
- Provides an introduction to the application and its functionalities.

### 2. **Scenario and Step Data Analysis**
- Analyze scenario execution durations and performance trends.
- Filter data by:
  - Specific scenarios (`scn_id`).
  - Date ranges.
- Aggregation methods:
  - Mean, Min, Max, and Median.
- Visualizations:
  - Line charts to track execution trends.
  - Side-by-side charts to compare different metrics.

### 3. **Status Data Analysis**
- Explore the distribution of scenario statuses (available/unavailable).
- Analyze temporal patterns in status changes.
- Visualizations:
  - Pie charts for status distribution.
  - Percentage bar charts for detailed insights.

### 4. **Prediction Integration**
- Display predictive results for scenario statuses and execution durations.
- Seamless integration with backend APIs to retrieve and visualize predictions.

## Data Structure
The application uses the following data components:

### 1. **Scenario Data**
- `scn_id`: Unique identifier for each scenario.
- Execution steps:
  - Each step has a recorded duration.

### 2. **Status Data**
- Binary status values:
  - `0`: Unavailable.
  - `1`: Available.

### 3. **Temporal Features**
- Extracted from the data such as Day of the week, Month, Part of Day ...

<!-- ## Setup and Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd streamlit-application
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run home.py
   ``` -->

<!-- ## Usage Instructions
1. Start the application and navigate through the pages using the sidebar.
2. On the **Scenario and Step Data Analysis** page:
   - Select a scenario and date range to view execution trends.
3. On the **Status Data Analysis** page:
   - View status distributions and analyze temporal patterns.
4. Visualize predictive results on the relevant pages. -->

## Media

### Video Demonstration
[![Watch the video](https://img.youtube.com/vi/IFdqrQ-Bj1Y/0.jpg)](https://www.youtube.com/watch?v=IFdqrQ-Bj1Y)



