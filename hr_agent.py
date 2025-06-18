import os
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMMathChain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import json
from datetime import datetime, timedelta
import pandas as pd
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QueryRequest(BaseModel):
    query: str
    user_context: Dict[str, Any]

class HRCalculator:
    @staticmethod
    def parse_numeric_input(input_str: str) -> float:
        """Parse numeric input from string, handling currency and percentages."""
        # Remove currency symbols and commas
        input_str = re.sub(r'[$,]', '', input_str)
        # Remove percentage sign and convert to decimal
        if '%' in input_str:
            return float(input_str.replace('%', '')) / 100
        return float(input_str)

    @staticmethod
    def calculate_leave_balance(join_date: str, total_leave: Union[str, float], used_leave: Union[str, float]) -> str:
        """Calculate remaining leave balance."""
        try:
            join_date = datetime.strptime(join_date, "%m/%d/%Y")
            total_leave = HRCalculator.parse_numeric_input(str(total_leave))
            used_leave = HRCalculator.parse_numeric_input(str(used_leave))
            remaining = total_leave - used_leave
            return f"Leave Balance: {remaining:.1f} days remaining out of {total_leave:.1f} total days"
        except Exception as e:
            return f"Error calculating leave balance: {str(e)}"

    @staticmethod
    def calculate_salary_with_raise(current_salary: Union[str, float], raise_percentage: Union[str, float]) -> str:
        """Calculate new salary after raise."""
        try:
            current_salary = HRCalculator.parse_numeric_input(str(current_salary))
            raise_percentage = HRCalculator.parse_numeric_input(str(raise_percentage))
            new_salary = current_salary * (1 + raise_percentage)
            return f"New Salary: ${new_salary:,.2f} (Current: ${current_salary:,.2f}, Raise: {raise_percentage*100:.1f}%)"
        except Exception as e:
            return f"Error calculating salary: {str(e)}"

    @staticmethod
    def calculate_working_days(start_date: str, end_date: str) -> str:
        """Calculate working days between two dates."""
        try:
            start = datetime.strptime(start_date, "%m/%d/%Y")
            end = datetime.strptime(end_date, "%m/%d/%Y")
            days = 0
            current = start
            while current <= end:
                if current.weekday() < 5:  # 0-4 are weekdays
                    days += 1
                current += timedelta(days=1)
            return f"Working Days: {days} days between {start_date} and {end_date}"
        except Exception as e:
            return f"Error calculating working days: {str(e)}"

    @staticmethod
    def calculate_benefits_cost(salary: Union[str, float], benefits_percentage: Union[str, float]) -> str:
        """Calculate benefits cost based on salary."""
        try:
            salary = HRCalculator.parse_numeric_input(str(salary))
            benefits_percentage = HRCalculator.parse_numeric_input(str(benefits_percentage))
            benefits_cost = salary * benefits_percentage
            return f"Benefits Cost: ${benefits_cost:,.2f} (Salary: ${salary:,.2f}, Benefits Rate: {benefits_percentage*100:.1f}%)"
        except Exception as e:
            return f"Error calculating benefits: {str(e)}"

    @staticmethod
    def parse_calculation_request(query: str) -> Dict[str, Any]:
        """Parse natural language query into calculation parameters."""
        try:
            # Extract dates (MM/DD/YYYY format)
            dates = re.findall(r'\d{2}/\d{2}/\d{4}', query)
            
            # Extract numbers (including currency and percentages)
            numbers = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?', query)
            numbers = [HRCalculator.parse_numeric_input(n) for n in numbers]
            
            # Extract keywords
            is_salary = any(word in query.lower() for word in ['salary', 'pay', 'income'])
            is_leave = any(word in query.lower() for word in ['leave', 'vacation', 'days off'])
            is_working_days = any(word in query.lower() for word in ['working days', 'business days'])
            is_benefits = any(word in query.lower() for word in ['benefits', 'benefit cost'])
            
            if is_salary and len(numbers) >= 2:
                return {
                    'type': 'salary',
                    'current_salary': numbers[0],
                    'raise_percentage': numbers[1]
                }
            elif is_leave and len(numbers) >= 2 and len(dates) >= 1:
                return {
                    'type': 'leave',
                    'join_date': dates[0],
                    'total_leave': numbers[0],
                    'used_leave': numbers[1]
                }
            elif is_working_days and len(dates) >= 2:
                return {
                    'type': 'working_days',
                    'start_date': dates[0],
                    'end_date': dates[1]
                }
            elif is_benefits and len(numbers) >= 2:
                return {
                    'type': 'benefits',
                    'salary': numbers[0],
                    'benefits_percentage': numbers[1]
                }
            
            return {'type': 'unknown', 'error': 'Could not parse calculation request'}
        except Exception as e:
            return {'type': 'error', 'error': str(e)}

class UserAuth:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.current_user: Optional[str] = None
        self.current_role: Optional[str] = None
        self.is_authenticated: bool = False
        # Special HR admin credentials
        self.hr_admin = {
            "username": "hr_admin",
            "password": "hrpassword123",
            "role": "hr"
        }
        # Log the DataFrame info for debugging
        logger.info(f"DataFrame loaded with {len(df)} rows")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"Sample employee_id values: {df['employee_id'].head().tolist()}")
        logger.info(f"Sample password values: {df['password'].head().tolist()}")

    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate user with employee ID and password, or HR admin credentials."""
        try:
            logger.info(f"Attempting authentication for username: {username}")
            
            # Check for HR admin login first
            if username == self.hr_admin["username"] and password == self.hr_admin["password"]:
                self.current_user = username
                self.current_role = self.hr_admin["role"]
                self.is_authenticated = True
                return True, f"Welcome HR Admin! You have full access to all employee data."

            # Convert username to string and strip any whitespace
            username = str(username).strip()
            password = str(password).strip()
            
            logger.info(f"Looking for employee_id: {username}")
            logger.info(f"DataFrame employee_id types: {self.df['employee_id'].dtype}")
            
            # Convert employee_id column to string for comparison
            self.df['employee_id'] = self.df['employee_id'].astype(str)
            
            # Find the user
            user_data = self.df[self.df['employee_id'] == username]
            logger.info(f"Found {len(user_data)} matching records")
            
            if len(user_data) == 0:
                logger.warning(f"No user found with employee_id: {username}")
                return False, "Invalid employee ID. Please check your ID and try again."
            
            # Log the found user's data (excluding password)
            user_info = user_data[['employee_id', 'name', 'position']].to_dict('records')[0]
            logger.info(f"Found user data: {user_info}")
            
            # Verify password
            stored_password = user_data['password'].values[0]
            logger.info(f"Comparing passwords - Input: {password}, Stored: {stored_password}")
            
            if stored_password != password:
                logger.warning(f"Password mismatch for user {username}")
                return False, "Incorrect password. Please try again."
            
            # Set user session
            self.current_user = user_data['name'].values[0]  # Store full name
            # Determine role based on position
            position = user_data['position'].values[0].lower()
            self.current_role = "hr" if "hr" in position else "employee"
            self.is_authenticated = True
            
            logger.info(f"Authentication successful for {self.current_user} as {self.current_role}")
            return True, f"Welcome {self.current_user}! You are logged in as {self.current_role}."
        
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            return False, "Authentication error. Please try again."

    def logout(self) -> str:
        """Clear user session."""
        if self.current_user:
            username = self.current_user
            self.current_user = None
            self.current_role = None
            self.is_authenticated = False
            return f"Goodbye {username}! You have been logged out."
        return "No active session to log out."

    def can_access_employee_data(self, target_employee: str) -> Tuple[bool, str]:
        """Check if current user can access target employee's data."""
        if not self.is_authenticated:
            return False, "Please log in to access employee data."
        
        if self.current_role == "hr" or self.current_user == "hr_admin":
            return True, "HR access granted."
        
        # Non-HR users can only access their own data
        if target_employee.lower() == self.current_user.lower():
            return True, "Access granted to own data."
        
        return False, f"Access denied. As a {self.current_role}, you can only access your own information."

class HRAgent:
    def employee_data_tool(self, query: str) -> str:
        """Execute a query on employee data with access control."""
        try:
            # Get current user info
            current_user = self.auth.current_user
            if not current_user:
                return "Error: Could not retrieve user information. Please try logging in again."
            
            # Debug logging
            self.logger.info(f"Current user: '{current_user}'")
            self.logger.info(f"Available names: {self.df['name'].tolist()}")
            
            current_user_name = current_user.strip()
            is_hr = self.auth.current_role == "hr"

            # Check if query is about another employee
            other_employee_mentioned = False
            mentioned_employee = None
            for _, row in self.df.iterrows():
                other_name = str(row['name']).strip()
                if other_name.lower() in query.lower() and other_name.lower() != current_user_name.lower():
                    other_employee_mentioned = True
                    mentioned_employee = other_name
                    break

            # If another employee is mentioned and user is not HR, deny access
            if other_employee_mentioned and not is_hr:
                return f"Access denied. You can only view your own information."

            # For self-referential queries or HR queries, proceed with the query
            try:
                # Replace "me", "my", "I" with the user's actual name
                processed_query = query
                for pronoun in ['me', 'my', 'i']:
                    if f" {pronoun} " in f" {processed_query.lower()} ":
                        processed_query = processed_query.replace(f" {pronoun} ", f" {current_user_name} ")
                        processed_query = processed_query.replace(f" {pronoun}'", f" {current_user_name}'")

                # If no name is mentioned, use current user's name
                if not any(str(row['name']).strip().lower() in processed_query.lower() for _, row in self.df.iterrows()):
                    if 'salary' in processed_query.lower() or 'benefits' in processed_query.lower():
                        # For salary/benefits queries, ensure we use the user's name
                        if not any(name.lower() in processed_query.lower() for name in [current_user_name.lower(), 'my', 'me', 'i']):
                            processed_query = f"What is {current_user_name}'s {processed_query}"

                # Debug logging
                self.logger.info(f"Processed query: '{processed_query}'")
                self.logger.info(f"Using name: '{current_user_name}'")

                # Get employee data
                employee_name = mentioned_employee if mentioned_employee else current_user_name
                employee_data = self.df[self.df['name'].str.strip().str.lower() == employee_name.strip().lower()]
                
                if len(employee_data) == 0:
                    return f"Could not find information for {employee_name}."

                # Get the first (and should be only) row
                row = employee_data.iloc[0]

                # Handle salary queries
                if 'salary' in processed_query.lower():
                    salary = float(row['basic_pay_in_php'])
                    if 'raise' in processed_query.lower() or 'increase' in processed_query.lower() or '%' in processed_query:
                        # Extract percentage if present
                        percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', processed_query)
                        if percentage_match:
                            percentage = float(percentage_match.group(1))
                            new_salary = salary * (1 + percentage/100)
                            return f"Current salary: ₱{salary:,.2f}\nWith {percentage}% raise: ₱{new_salary:,.2f}"
                        return f"Current salary: ₱{salary:,.2f}"
                    return f"Current salary: ₱{salary:,.2f}"

                # Handle benefits queries
                if 'benefits' in processed_query.lower():
                    salary = float(row['basic_pay_in_php'])
                    # Extract percentage if present
                    percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', processed_query)
                    if percentage_match:
                        percentage = float(percentage_match.group(1))
                        benefits_cost = salary * (percentage/100)
                        return f"Benefits cost at {percentage}% rate: ₱{benefits_cost:,.2f}"
                    return f"Please specify the benefits rate percentage."

                # Handle leave queries
                if 'leave' in processed_query.lower():
                    if 'vacation' in processed_query.lower():
                        leave_days = int(row['vacation_leave'])
                        return f"Vacation leave balance: {leave_days} days"
                    elif 'sick' in processed_query.lower():
                        leave_days = int(row['sick_leave'])
                        return f"Sick leave balance: {leave_days} days"
                    else:
                        vacation = int(row['vacation_leave'])
                        sick = int(row['sick_leave'])
                        return f"Leave balances:\nVacation leave: {vacation} days\nSick leave: {sick} days"

                # Handle role/position queries
                if 'role' in processed_query.lower() or 'position' in processed_query.lower():
                    return f"Position: {row['position']}"

                # Handle other employee data queries
                if 'rank' in processed_query.lower():
                    return f"Rank: {row['rank']}"
                if 'hire date' in processed_query.lower():
                    return f"Hire date: {row['hire_date']}"
                if 'regularization' in processed_query.lower():
                    return f"Regularization date: {row['regularization_date']}"
                if 'status' in processed_query.lower():
                    return f"Employment status: {row['employment_status']}"

                # If no specific query matched, return basic info
                return f"Employee Information for {employee_name}:\n" + \
                       f"Position: {row['position']}\n" + \
                       f"Rank: {row['rank']}\n" + \
                       f"Basic Pay: ₱{float(row['basic_pay_in_php']):,.2f}\n" + \
                       f"Vacation Leave: {int(row['vacation_leave'])} days\n" + \
                       f"Sick Leave: {int(row['sick_leave'])} days"

            except Exception as e:
                self.logger.error(f"Error querying employee data: {str(e)}")
                return f"Error querying employee data: {str(e)}"

        except Exception as e:
            self.logger.error(f"Error in employee_data_tool: {str(e)}")
            return f"Error in employee_data_tool: {str(e)}"

    def load_policies_from_file(self, file_path: str = "hr_policy.txt") -> Dict[str, str]:
        """
        Load policies from the existing hr_policy.txt file with alphabetical section markers.
        
        Args:
            file_path: Path to the policy file (default: hr_policy.txt)
            
        Returns:
            Dictionary with policy names as keys and content as values
        """
        try:
            policies = {}
            current_section = None
            current_content = []
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.logger.error(f"Policy file not found: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split content by major sections
                sections = content.split('\n\n')
                
                for section in sections:
                    lines = section.strip().split('\n')
                    if not lines:
                        continue
                    
                    # Check for section headers (A., B., C., etc.)
                    first_line = lines[0].strip()
                    if re.match(r'^[A-Z]\.\s+', first_line):
                        # Save previous section if exists
                        if current_section and current_content:
                            policies[current_section] = '\n'.join(current_content).strip()
                        
                        # Extract section name (remove the letter and dot)
                        section_name = first_line.split('.', 1)[1].strip()
                        current_section = section_name.lower().replace(' ', '_')
                        current_content = [first_line] + lines[1:]
                    elif current_section:
                        # Add content to current section
                        current_content.extend(lines)
                
                # Save the last section
                if current_section and current_content:
                    policies[current_section] = '\n'.join(current_content).strip()
            
            # Add special sections for common queries
            if 'vacation_leave' not in policies and 'A. Vacation Leave' in content:
                # Extract vacation leave section
                vacation_match = re.search(r'A\.\s+Vacation Leave.*?(?=B\.|$)', content, re.DOTALL)
                if vacation_match:
                    policies['vacation_leave'] = vacation_match.group(0).strip()
            
            if 'sick_leave' not in policies and 'B. Sick Leave' in content:
                # Extract sick leave section
                sick_match = re.search(r'B\.\s+Sick Leave.*?(?=C\.|$)', content, re.DOTALL)
                if sick_match:
                    policies['sick_leave'] = sick_match.group(0).strip()
            
            if 'leave_encashment' not in policies and 'VI. LEAVE ENCASHMENT' in content:
                # Extract leave encashment section
                encashment_match = re.search(r'VI\.\s+LEAVE ENCASHMENT.*?(?=VII\.|$)', content, re.DOTALL)
                if encashment_match:
                    policies['leave_encashment'] = encashment_match.group(0).strip()
            
            # Add overtime section if found
            if 'Overtime' in content:
                overtime_match = re.search(r'B\.\s+Overtime.*?(?=C\.|$)', content, re.DOTALL)
                if overtime_match:
                    policies['overtime'] = overtime_match.group(0).strip()
            
            # Add work from home section if found
            if 'Work From Home' in content:
                wfh_match = re.search(r'D\.\s+Work From Home.*?(?=E\.|$)', content, re.DOTALL)
                if wfh_match:
                    policies['work_from_home'] = wfh_match.group(0).strip()
            
            self.logger.info(f"Successfully loaded {len(policies)} policy sections from {file_path}")
            return policies
            
        except Exception as e:
            self.logger.error(f"Error loading policies from file {file_path}: {str(e)}")
            return {}

    def policy_tool(self, query: str) -> str:
        """Handle policy-related queries."""
        try:
            # Use pre-loaded policies from initialization
            policies = self.policies
            
            if not policies:
                return "Error: Could not load policy information. Please contact HR department."
            
            # Determine which policy to return based on query
            query_lower = query.lower()
            
            # Keyword mapping for policy detection
            if any(term in query_lower for term in ['encash', 'encashment', 'monetize', 'convert to money']):
                if 'leave_encashment' in policies:
                    return policies['leave_encashment']
                else:
                    return "Leave encashment policy information is not available."
                    
            elif any(term in query_lower for term in ['vacation', 'annual leave', 'paid leave']):
                if 'vacation_leave' in policies:
                    return policies['vacation_leave']
                else:
                    return "Vacation leave policy information is not available."
                    
            elif any(term in query_lower for term in ['sick', 'medical leave', 'illness']):
                if 'sick_leave' in policies:
                    return policies['sick_leave']
                else:
                    return "Sick leave policy information is not available."
                    
            elif any(term in query_lower for term in ['overtime', 'extra hours', 'additional work']):
                if 'overtime' in policies:
                    return policies['overtime']
                else:
                    return "Overtime policy information is not available."
                    
            elif any(term in query_lower for term in ['work from home', 'remote work', 'wfh']):
                if 'work_from_home' in policies:
                    return policies['work_from_home']
                else:
                    return "Work from home policy information is not available."
                    
            elif any(term in query_lower for term in ['benefits', 'benefit', 'insurance', 'coverage']):
                if 'benefits' in policies:
                    return policies['benefits']
                else:
                    return "Benefits policy information is not available."
            else:
                # Return available policy sections if no specific match
                available_policies = list(policies.keys())
                return f"I'm not sure about the specific policy. Available policies include: {', '.join(available_policies)}. Please try rephrasing your question about one of these topics."

        except Exception as e:
            self.logger.error(f"Error in policy tool: {str(e)}")
            return f"Error retrieving policy information: {str(e)}"

    def calculator_tool(self, query: str) -> str:
        """Handle calculation-related queries."""
        try:
            # Get employee data if needed
            current_user = self.auth.current_user
            if not current_user:
                return "Error: Could not retrieve user information."
            
            # Handle different types of calculations
            if 'encash' in query.lower():
                # Get employee's salary and leave balance
                employee_data = self.df[self.df['name'].str.strip().str.lower() == current_user.strip().lower()]
                if len(employee_data) == 0:
                    return f"Could not find information for {current_user}."
                
                row = employee_data.iloc[0]
                salary = float(row['basic_pay_in_php'])
                leave_days = int(row['vacation_leave'])
                
                # Calculate encashment (salary/30 * days * 1.25)
                daily_rate = salary / 30
                encashment = daily_rate * leave_days * 1.25
                
                return f"""
                Leave Encashment Calculation for {current_user}:
                - Current Salary: ₱{salary:,.2f}
                - Available Leave Days: {leave_days}
                - Daily Rate: ₱{daily_rate:,.2f}
                - Encashment Amount: ₱{encashment:,.2f}
                
                Note: This is based on the maximum allowed encashment of 50% of unused leave.
                """
            
            elif 'benefit' in query.lower():
                # Extract percentage if present
                percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', query)
                if not percentage_match:
                    return "Please specify the benefits rate percentage."
                
                percentage = float(percentage_match.group(1))
                employee_data = self.df[self.df['name'].str.strip().str.lower() == current_user.strip().lower()]
                if len(employee_data) == 0:
                    return f"Could not find information for {current_user}."
                
                salary = float(employee_data.iloc[0]['basic_pay_in_php'])
                benefits_cost = salary * (percentage/100)
                
                return f"""
                Benefits Calculation for {current_user}:
                - Current Salary: ₱{salary:,.2f}
                - Benefits Rate: {percentage}%
                - Benefits Cost: ₱{benefits_cost:,.2f}
                """
            
            else:
                return "I can help calculate leave encashment or benefits. Please specify which calculation you need."

        except Exception as e:
            self.logger.error(f"Error in calculator tool: {str(e)}")
            return f"Error performing calculation: {str(e)}"

    def __init__(self):
        """Initialize the HR agent with necessary tools and authentication."""
        try:
            # Initialize logger
            self.logger = logging.getLogger(__name__)
            
            # Load employee data
            self.df = pd.read_csv('sample_employee.csv')
            self.df['employee_id'] = self.df['employee_id'].astype(str)
            
            # Initialize authentication
            self.auth = UserAuth(self.df)
            
            # Load policies once during initialization
            self.policies = self.load_policies_from_file("hr_policy.txt")
            if not self.policies:
                self.logger.warning("No policies loaded from file. Policy tool may not function properly.")
            else:
                self.logger.info(f"Loaded {len(self.policies)} policy sections")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Get column names for tool description
            df_columns = ', '.join(self.df.columns.tolist())
            
            # Initialize tools
            self.employee_data_tool = Tool(
                name="Employee Data",
                func=self.employee_data_tool,
                description=f"""
                Useful for when you need to answer questions about employee data stored in pandas dataframe.
                The dataframe has the following columns: {df_columns}
                
                Examples:
                - How many sick leave days do I have left?
                - What is my current salary?
                - What is my position?
                - What is my hire date?
                
                For policy-related questions (like leave encashment, carryover rules, etc.), use the Policy tool instead.
                """
            )
            
            self.policy_tool = Tool(
                name="HR Policies",
                func=self.policy_tool,
                description="""
                Useful for answering questions about company policies and procedures.
                
                Examples:
                - What is the policy on unused vacation leave?
                - How does leave encashment work?
                - What happens to unused sick leave at year end?
                - What is the overtime policy?
                - How are benefits calculated?
                - What is the policy on work from home?
                
                This tool should be used for any questions about rules, procedures, or policies.
                """
            )
            
            self.calculator_tool = Tool(
                name="Calculator",
                func=self.calculator_tool,
                description="""
                Useful for performing calculations related to employee data.
                
                Examples:
                - Calculate benefits cost with a 15% rate
                - Calculate new salary after a 10% raise
                - Calculate leave encashment amount
                - Calculate overtime pay
                
                Always use this tool for any mathematical operations.
                """
            )
            
            # Initialize the agent with all tools
            self.agent = initialize_agent(
                [self.employee_data_tool, self.policy_tool, self.calculator_tool],
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
            
            self.logger.info("Agent tools setup complete")
            self.logger.info("Initializing Langchain agent...")
            self.logger.info("Langchain agent initialized successfully")
            self.logger.info("HR Agent initialization complete")
            
        except Exception as e:
            self.logger.error(f"Error initializing HR agent: {str(e)}")
            raise

    def get_response(self, query: str, user_context: Dict[str, Any]) -> str:
        """Get a response from the HR agent for a user query."""
        try:
            # Validate user access first
            if not self.validate_user_access(user_context, query):
                return "Access denied. You do not have permission to access this information."

            # Get current user info
            current_user = self.auth.current_user
            if not current_user:
                return "Error: Could not retrieve user information. Please try logging in again."
            
            # Handle authentication-related queries separately
            auth_keywords = ['login', 'logout', 'authenticate', 'password']
            if any(keyword in query.lower() for keyword in auth_keywords):
                return "Please use the login/logout functionality in the interface for authentication-related requests."

            # Use the agent to determine which tool to use
            try:
                response = self.agent.invoke({"input": query})
                if response and isinstance(response, dict) and "output" in response:
                    return response["output"]
                return "I'm not sure how to help with that. Please try rephrasing your question."
            except Exception as e:
                self.logger.error(f"Error getting agent response: {str(e)}")
                return "I apologize, but I encountered an error processing your request. Please try rephrasing your question."

        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def validate_user_access(self, user_context: Dict[str, Any], query: str) -> bool:
        """
        Validate if the user has access to make the query.
        
        Args:
            user_context: Dictionary containing user context information
            query: The user's query string
            
        Returns:
            bool: True if access is granted, False otherwise
        """
        try:
            # Check if user is authenticated
            if not self.auth.is_authenticated:
                self.logger.warning(f"Access denied: User not authenticated")
                return False
            
            # Get current user and role information
            current_user = self.auth.current_user
            current_role = self.auth.current_role
            
            if not current_user:
                self.logger.warning(f"Access denied: Could not retrieve user information")
                return False
            
            # HR users have full access to all queries
            if current_role == "hr":
                self.logger.info(f"Access granted: HR user '{current_user}' has full access")
                return True
            
            query_lower = query.lower()
            current_user_lower = current_user.lower()
            
            # Check for other employee names in the query (deny if found)
            for _, row in self.df.iterrows():
                other_employee = str(row['name']).strip().lower()
                if other_employee != current_user_lower and other_employee in query_lower:
                    self.logger.warning(f"Access denied: Non-HR user '{current_user}' attempted to access data for '{row['name']}'")
                    return False
            
            # Check if query is self-referential (contains user's name or pronouns)
            if current_user_lower in query_lower:
                self.logger.info(f"Access granted: Self-referential query for user '{current_user}'")
                return True
            
            self_referential_terms = [
                'my salary', 'my leave', 'my benefits', 'my information',
                'my vacation', 'my sick leave', 'my position', 'my hire date',
                'my employment', 'my data'
            ]
            if any(term in query_lower for term in self_referential_terms):
                self.logger.info(f"Access granted: Self-referential query for user '{current_user}'")
                return True
            standalone_self_terms = ['my', 'me', 'i', 'myself', 'mine']
            words = query_lower.split()
            if any(term in words for term in standalone_self_terms):
                # Additional check: make sure it's not about someone else
                if not any(str(row['name']).strip().lower() in query_lower for _, row in self.df.iterrows()):
                    self.logger.info(f"Access granted: Self-referential query for user '{current_user}'")
                    return True
            
            # Check if query is about policy information (generally accessible)
            policy_terms = [
                'policy', 'policies', 'encashment', 'leave', 'overtime', 
                'work from home', 'benefits', 'rules', 'procedures'
            ]
            if any(term in query_lower for term in policy_terms):
                self.logger.info(f"Access granted: Policy-related query for user '{current_user}'")
                return True
            
            # Default: deny access for non-HR users for all other queries
            self.logger.warning(f"Access denied: Query did not match any allowed criteria for user '{current_user}'")
            return False
        except Exception as e:
            self.logger.error(f"Error in validate_user_access: {str(e)}")
            # Default to deny access on error for security
            return False