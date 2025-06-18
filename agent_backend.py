from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from hr_agent import HRAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = {
    "OPENAI_API_KEY": "OpenAI API key for embeddings and chat",
    "PINECONE_API_KEY": "Pinecone API key for vector storage",
    "PINECONE_ENVIRONMENT": "Pinecone environment (e.g., gcp-starter)",
    "PINECONE_INDEX_NAME": "Name of the Pinecone index",
    "JWT_SECRET_KEY": "Secret key for JWT token generation"
}

missing_vars = [var for var, desc in required_env_vars.items() if not os.getenv(var)]
if missing_vars:
    error_msg = "Missing required environment variables:\n" + "\n".join(
        f"- {var}: {required_env_vars[var]}" for var in missing_vars
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

# Initialize FastAPI app
app = FastAPI(title="HR Agent API", description="API for HR Assistant Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize HR Agent
hr_agent = HRAgent()

# Models
class User(BaseModel):
    username: str
    email: Optional[str] = None
    role: str
    department: Optional[str] = None
    employee_id: Optional[str] = None
    name: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Token(BaseModel):
    access_token: str
    token_type: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ChatRequest(BaseModel):
    message: str
    user_context: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ChatResponse(BaseModel):
    response: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Helper functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    # Create user object from token data
    user = User(
        username=token_data.username,
        role=token_data.role,
        name=hr_agent.auth.current_user if hr_agent.auth.is_authenticated else None
    )
    return user

# API Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Use HR agent's authentication
    success, message = hr_agent.auth.authenticate(form_data.username, form_data.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": form_data.username,
            "role": hr_agent.auth.current_role
        },
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Use user context from request if available, otherwise use current_user
        user_context = request.user_context or {
            "username": current_user.username,
            "name": current_user.name,
            "role": current_user.role,
            "employee_id": current_user.employee_id
        }
        
        # Validate user access
        if not hr_agent.validate_user_access(user_context, request.message):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this information"
            )
        
        # Get response from agent with proper error handling
        try:
            response = hr_agent.get_response(request.message, user_context)
            if not response or response.strip() == "":
                raise ValueError("Empty response from agent")
            return ChatResponse(response=response)
        except Exception as agent_error:
            logger.error(f"Agent error: {str(agent_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing your request: {str(agent_error)}"
            )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )

@app.get("/user/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    if hr_agent.auth.is_authenticated:
        try:
            # Get user data from employee database
            user_data = hr_agent.df[hr_agent.df['employee_id'] == current_user.username]
            if not user_data.empty:
                # Update user info with data from database
                current_user.name = user_data['name'].values[0]
                current_user.role = hr_agent.auth.current_role
                current_user.department = user_data['organizational_unit'].values[0]
                current_user.employee_id = str(user_data['employee_id'].values[0])
            else:
                logger.error(f"Could not find user data for employee_id: {current_user.username}")
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
    return current_user

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
