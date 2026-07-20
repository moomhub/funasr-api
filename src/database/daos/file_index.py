"""Data access operations for archived-file indexes."""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import S3File


class S3FileDAO:
    """Persist S3 and local archive index records."""

    @staticmethod
    def get_by_hash(file_sha256: str, *, session: Session) -> Optional[S3File]:
        return session.query(S3File).filter(S3File.file_sha256 == file_sha256).first()

    @staticmethod
    def create(
        task_key: str,
        task_type: str,
        storage_backend: str,
        s3_key: str,
        stored_filename: str,
        file_sha256: str,
        original_filename: str = None,
        bucket_name: str = None,
        file_size: int = None,
        content_type: str = None,
        local_path: str = None,
        upload_status: str = "uploaded",
        is_reused: bool = False,
        *,
        session: Session,
    ) -> S3File:
        item = S3File(
            task_key=task_key,
            task_type=task_type,
            storage_backend=storage_backend,
            bucket_name=bucket_name,
            s3_key=s3_key,
            stored_filename=stored_filename,
            original_filename=original_filename,
            file_sha256=file_sha256,
            file_size=file_size,
            content_type=content_type,
            local_path=local_path,
            upload_status=upload_status,
            is_reused=is_reused,
        )
        session.add(item)
        session.flush()
        return item


__all__ = ["S3FileDAO"]
