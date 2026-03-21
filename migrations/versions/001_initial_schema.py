"""Initial schema: logs, templates, cluster_results.

Revision ID: 001
Revises: None
Create Date: 2026-03-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Templates (must be created before logs due to FK)
    op.create_table(
        "templates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("microservice", sa.String(128), nullable=False),
        sa.Column("drain_cluster_id", sa.Integer(), nullable=False),
        sa.Column("template_text", sa.Text(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.Column("stacktrace_pattern", sa.String(256), nullable=True),
        sa.Column(
            "first_seen",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("log_count", sa.Integer(), nullable=False, server_default="1"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_templates_service_drain",
        "templates",
        ["microservice", "drain_cluster_id"],
        unique=True,
    )

    # Logs
    op.create_table(
        "logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("microservice", sa.String(128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("level", sa.String(16), nullable=False),
        sa.Column("raw_message", sa.Text(), nullable=False),
        sa.Column(
            "host", sa.String(128), nullable=False, server_default="unknown"
        ),
        sa.Column("stacktrace", sa.Text(), nullable=True),
        sa.Column(
            "template_id",
            sa.Integer(),
            sa.ForeignKey("templates.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_logs_service_timestamp", "logs", ["microservice", "timestamp"]
    )
    op.create_index("ix_logs_template_id", "logs", ["template_id"])

    # Cluster results
    op.create_table(
        "cluster_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("microservice", sa.String(128), nullable=False),
        sa.Column("period", sa.String(16), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("num_clusters", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("noise_ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("silhouette_score", sa.Float(), nullable=True),
        sa.Column("result_data", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cluster_results_lookup",
        "cluster_results",
        ["microservice", "period", "computed_at"],
    )


def downgrade() -> None:
    op.drop_table("cluster_results")
    op.drop_table("logs")
    op.drop_table("templates")
