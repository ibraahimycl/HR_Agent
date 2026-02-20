from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from hr_agent import HRAgent
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate only essential env vars
required_env_vars = {
    "OPENAI_API_KEY": "OpenAI API key for embeddings and chat",
    "JWT_SECRET_KEY": "Secret key for JWT token generation",
}
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        "Missing required environment variables:\n"
        + "\n".join(f"- {v}: {required_env_vars[v]}" for v in missing_vars)
    )

app = FastAPI(title="HR Agent API", description="API for HR Assistant Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

hr_agent = HRAgent()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


class ChatRequest(BaseModel):
    message: str
    user_context: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentStep(BaseModel):
    tool: str
    input: Any
    output: str


class ChatResponse(BaseModel):
    response: str
    steps: List[AgentStep] = []
    citations: List[Dict[str, Any]] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or not role:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    return User(username=username, role=role)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    success, message = hr_agent.auth.authenticate(form_data.username, form_data.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": form_data.username, "role": hr_agent.auth.current_role},
        expires_delta=expires,
    )
    return {"access_token": token, "token_type": "bearer"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        user_context = request.user_context or {
            "username": current_user.username,
            "name": current_user.name or current_user.username,
            "role": current_user.role,
            "employee_id": current_user.employee_id,
        }

        if not hr_agent.validate_user_access(user_context, request.message):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied.",
            )

        result = hr_agent.get_response(request.message, user_context)

        steps = [
            AgentStep(
                tool=s["tool"],
                input=s["input"],
                output=str(s["output"]),
            )
            for s in result.get("steps", [])
        ]

        return ChatResponse(
            response=result["answer"],
            steps=steps,
            citations=result.get("citations", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/user/me", response_model=User)
async def read_me(current_user: User = Depends(get_current_user)):
    try:
        hr_agent.df['employee_id'] = hr_agent.df['employee_id'].astype(str)
        row = hr_agent.df[hr_agent.df['employee_id'] == current_user.username]
        if not row.empty:
            current_user.name = str(row['name'].values[0])
            current_user.role = hr_agent.auth.current_role or current_user.role
            current_user.department = str(row['organizational_unit'].values[0])
            current_user.employee_id = str(row['employee_id'].values[0])
    except Exception as e:
        logger.error(f"Error loading user data: {e}")
    return current_user


@app.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    hr_agent.clear_session(current_user.username)
    return {"message": "Session cleared."}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
