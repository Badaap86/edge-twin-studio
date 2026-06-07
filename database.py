# database.py
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime
import uuid

# --- CONFIGURATIE ---
# Voor lokaal testen gebruiken we nog even SQLite, 
# maar voor productie verander je deze string naar je PostgreSQL URL (bijv. Supabase/Neon).
DATABASE_URL = "sqlite:///./omega_saas.db"
# DATABASE_URL = "postgresql://user:password@localhost/omega_db" 

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

# ============================================================
# SAAS DATA MODELLEN (MULTI-TENANCY)
# ============================================================

class Organization(Base):
    """Bedrijven die betalen voor jouw platform (bijv. 'Philips', 'Siemens')"""
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relaties
    users = relationship("User", back_populates="organization")
    projects = relationship("Project", back_populates="organization")

class User(Base):
    """De individuele gebruikers (engineers/admins) binnen een organisatie"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="engineer") # 'admin', 'engineer', 'viewer'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relaties
    organization = relationship("Organization", back_populates="users")

class Project(Base):
    """De daadwerkelijke Edge AI projecten, gekoppeld aan een organisatie"""
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=generate_uuid)
    org_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    name = Column(String, index=True, nullable=False)
    dataset = Column(Text, nullable=True)  # Opgeslagen als JSON string
    settings = Column(JSON, nullable=True) # DSP Parameters, Base Freq, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relaties
    organization = relationship("Organization", back_populates="projects")

# ============================================================
# DATABASE INITIALISATIE
# ============================================================
def init_db():
    """Maakt de tabellen aan als ze nog niet bestaan."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Helper voor FastAPI om database-sessies te beheren."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
