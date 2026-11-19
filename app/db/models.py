"""Authoritative relational schemas for documents, pages, assets, chunks, chat."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    document_type: Mapped[str] = mapped_column(String(32), default="unknown")  # A|B|C|D
    document_category: Mapped[str] = mapped_column(String(128), default="general")
    document_version: Mapped[str] = mapped_column(String(64), default="")
    storage_path: Mapped[str] = mapped_column(String(1024), default="")
    processing_status: Mapped[str] = mapped_column(
        String(32), default="pending", index=True
    )  # pending|processing|partial|completed|failed
    access_scope: Mapped[str] = mapped_column(String(128), default="internal")
    access_permissions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, default=dict)
    uploaded_by: Mapped[str] = mapped_column(String(128), default="system")
    model_versions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, default=dict)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    pages: Mapped[list[PageManifest]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list[ChunkRecord]] = relationship(back_populates="document", cascade="all, delete-orphan")
    occurrences: Mapped[list[ImageOccurrence]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    reports: Mapped[list[IngestionReport]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class PageManifest(Base):
    __tablename__ = "page_manifests"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_doc_page"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    page_type: Mapped[str] = mapped_column(String(32), default="unknown")
    # text | image | text_image | scanned | table | diagram | empty | mixed
    text_extraction_status: Mapped[str] = mapped_column(String(32), default="pending")
    ocr_status: Mapped[str] = mapped_column(String(32), default="pending")
    image_extraction_status: Mapped[str] = mapped_column(String(32), default="pending")
    table_extraction_status: Mapped[str] = mapped_column(String(32), default="pending")
    diagram_analysis_status: Mapped[str] = mapped_column(String(32), default="pending")
    duplicate_detection_status: Mapped[str] = mapped_column(String(32), default="pending")
    chunking_status: Mapped[str] = mapped_column(String(32), default="pending")
    embedding_status: Mapped[str] = mapped_column(String(32), default="pending")
    indexing_status: Mapped[str] = mapped_column(String(32), default="pending")
    processing_errors: Mapped[Optional[list[Any]]] = mapped_column(JSON, default=list)
    extracted_content_count: Mapped[int] = mapped_column(Integer, default=0)
    outcome: Mapped[str] = mapped_column(String(64), default="pending")
    # text_ok | image_ok | ocr_ok | empty | duplicate_referenced | failed
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    document: Mapped[Document] = relationship(back_populates="pages")


class ImageAsset(Base):
    __tablename__ = "image_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    perceptual_hash: Mapped[str] = mapped_column(String(32), default="", index=True)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    image_format: Mapped[str] = mapped_column(String(16), default="png")
    image_type: Mapped[str] = mapped_column(String(64), default="content")
    image_description: Mapped[str] = mapped_column(Text, default="")
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    embedding_ref: Mapped[str] = mapped_column(String(128), default="")
    storage_status: Mapped[str] = mapped_column(String(32), default="stored")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    occurrences: Mapped[list[ImageOccurrence]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class ImageOccurrence(Base):
    __tablename__ = "image_occurrences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    asset_id: Mapped[str] = mapped_column(ForeignKey("image_assets.id"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, default=dict)
    image_role: Mapped[str] = mapped_column(String(64), default="content")
    # content | header_footer | watermark | duplicate_occurrence
    indexable: Mapped[bool] = mapped_column(Boolean, default=True)
    caption: Mapped[str] = mapped_column(Text, default="")
    associated_chunk_ids: Mapped[Optional[list[Any]]] = mapped_column(JSON, default=list)
    section_heading: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    asset: Mapped[ImageAsset] = relationship(back_populates="occurrences")
    document: Mapped[Document] = relationship(back_populates="occurrences")


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer, default=0)
    section_heading: Mapped[str] = mapped_column(String(512), default="")
    parent_section: Mapped[str] = mapped_column(String(512), default="")
    chunk_type: Mapped[str] = mapped_column(String(64), default="text")
    chunk_text: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    previous_chunk_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    next_chunk_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    associated_image_ids: Mapped[Optional[list[Any]]] = mapped_column(JSON, default=list)
    associated_table_ids: Mapped[Optional[list[Any]]] = mapped_column(JSON, default=list)
    source_coordinates: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, default=dict)
    okf_path: Mapped[str] = mapped_column(String(1024), default="")
    indexable: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class IngestionReport(Base):
    __tablename__ = "ingestion_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    complete: Mapped[bool] = mapped_column(Boolean, default=False)
    pages_total: Mapped[int] = mapped_column(Integer, default=0)
    pages_processed: Mapped[int] = mapped_column(Integer, default=0)
    pages_failed: Mapped[Optional[list[Any]]] = mapped_column(JSON, default=list)
    summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="reports")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), default="anonymous", index=True)
    title: Mapped[str] = mapped_column(String(512), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class AnswerFeedback(Base):
    """Thumbs up/down feedback on assistant answers (Phase 1 MVP)."""

    __tablename__ = "answer_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    message_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    user_id: Mapped[str] = mapped_column(String(128), default="anonymous", index=True)
    query: Mapped[str] = mapped_column(Text, default="")
    rating: Mapped[str] = mapped_column(String(8), nullable=False)  # up | down
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
