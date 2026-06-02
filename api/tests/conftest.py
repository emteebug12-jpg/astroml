"""
Shared pytest fixtures for API integration tests.

Issue #244 — Integration tests for API endpoints and CI pipeline.
Issue #246 — Database Session & Models.

Provides:
  - `db_engine`             : ephemeral SQLite engine per test function
  - `db_session`            : isolated session (rolled back on teardown)
  - `client`                : FastAPI TestClient with DB override
  - `seeded_account`        : one Account row in the test DB
  - `seeded_transaction`    : one Transaction row in the test DB
  - `seeded_alert`          : one FraudAlert row in the test DB
  - `seeded_loyalty`        : one LoyaltyPoints row in the test DB
  - `sample_accounts`       : raw list-of-dicts (no DB) for unit-level tests
  - `sample_transactions`   : raw list-of-dicts
  - `sample_alerts`         : raw list-of-dicts
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from astroml.db.schema import Base
import api.models.orm  # noqa: F401 — registers ORM models on Base.metadata
from api.models.orm import Account, FraudAlert, LoyaltyPoints, PointsTransaction, Transaction

# ─── Engine / session ─────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_engine(tmp_path):
    """Ephemeral SQLite engine scoped to this test function."""
    db_file = tmp_path / "test_astroml.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_wal(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Session:
    """Clean session per test — all writes are rolled back on teardown."""
    SessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


# ─── FastAPI TestClient with DB override ──────────────────────────────────────

@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with the real DB dependency replaced by the test session."""
    from api.app import app
    from api.database import get_sync_db

    def _override_db():
        yield db_session

    app.dependency_overrides[get_sync_db] = _override_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ─── ORM seed fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def seeded_account(db_session) -> Account:
    acc = Account(
        public_key="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_active=datetime(2024, 6, 1, tzinfo=timezone.utc),
        balance=1000.0,
    )
    db_session.add(acc)
    db_session.flush()
    return acc


@pytest.fixture()
def seeded_transaction(db_session) -> Transaction:
    tx = Transaction(
        hash="abc123def456abc123def456abc123def456abc123def456abc123def456ab12",
        ledger_sequence=100,
        source_account="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        destination_account="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
        amount=500.0,
        asset_code="XLM",
        fee=100,
        successful=True,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    db_session.add(tx)
    db_session.flush()
    return tx


@pytest.fixture()
def seeded_alert(db_session) -> FraudAlert:
    alert = FraudAlert(
        account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        pattern="sybil_cluster",
        risk_score=0.92,
        risk_level="high",
        description="Suspicious transaction velocity detected.",
        detected_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    db_session.add(alert)
    db_session.flush()
    return alert


@pytest.fixture()
def seeded_loyalty(db_session) -> LoyaltyPoints:
    lp = LoyaltyPoints(
        account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        balance=2500,
        tier="silver",
        multiplier=1.1,
    )
    db_session.add(lp)
    db_session.flush()
    return lp


# ─── Raw dict fixtures (no DB, unit-level) ────────────────────────────────────

@pytest.fixture()
def sample_accounts():
    return [
        {"account_id": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN", "sequence": 1},
        {"account_id": "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT", "sequence": 2},
        {"account_id": "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF", "sequence": 3},
    ]


@pytest.fixture()
def sample_transactions():
    return [
        {
            "transaction_hash": "abc123",
            "ledger_sequence": 100,
            "source_account": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "fee_charged": 100,
            "operation_count": 1,
            "successful": True,
        },
        {
            "transaction_hash": "def456",
            "ledger_sequence": 101,
            "source_account": "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            "fee_charged": 200,
            "operation_count": 2,
            "successful": True,
        },
        {
            "transaction_hash": "ghi789",
            "ledger_sequence": 102,
            "source_account": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "fee_charged": 100,
            "operation_count": 1,
            "successful": False,
        },
    ]


@pytest.fixture()
def sample_alerts():
    return [
        {
            "alert_id": "a1",
            "account_id": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "severity": "high",
            "resolved": False,
        },
        {
            "alert_id": "a2",
            "account_id": "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            "severity": "low",
            "resolved": True,
        },
    ]
