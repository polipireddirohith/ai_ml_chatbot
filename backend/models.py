from sqlalchemy import Column, Integer, String, Float, Text
import database


class ChatLog(database.Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, index=True)
    response = Column(Text)
    model_used = Column(String)
    timestamp = Column(Float)
