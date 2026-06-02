"""
Integration tests — loyalty points (issue #244).

Covers: ORM model creation, tier logic, points transactions,
balance queries, and history pagination.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from api.models.orm import LoyaltyPoints, PointsTransaction


@pytest.mark.xdist_group("api_loyalty")
class TestLoyaltyPointsModel:
    """ORM-level tests for LoyaltyPoints (issue #246)."""

    def test_create_loyalty_row(self, db_session):
        lp = LoyaltyPoints(
            account_id="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            balance=500,
            tier="bronze",
            multiplier=1.0,
        )
        db_session.add(lp)
        db_session.flush()
        assert lp.id is not None

    def test_seeded_loyalty_fields(self, seeded_loyalty):
        assert seeded_loyalty.balance == 2500
        assert seeded_loyalty.tier == "silver"
        assert seeded_loyalty.multiplier == pytest.approx(1.1)

    def test_unique_account_constraint(self, db_session, seeded_loyalty):
        duplicate = LoyaltyPoints(
            account_id=seeded_loyalty.account_id,
            balance=0,
            tier="bronze",
            multiplier=1.0,
        )
        db_session.add(duplicate)
        with pytest.raises(Exception):
            db_session.flush()

    def test_query_by_account(self, db_session, seeded_loyalty):
        result = db_session.execute(
            select(LoyaltyPoints).where(
                LoyaltyPoints.account_id == seeded_loyalty.account_id
            )
        ).scalar_one()
        assert result.balance == 2500


@pytest.mark.xdist_group("api_loyalty")
class TestPointsTransactionModel:
    """ORM-level tests for PointsTransaction (issue #246)."""

    def test_create_earn_transaction(self, db_session):
        pt = PointsTransaction(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            type="earn",
            points=100,
            source="trade_completion",
        )
        db_session.add(pt)
        db_session.flush()
        assert pt.id is not None

    def test_create_redeem_transaction(self, db_session):
        pt = PointsTransaction(
            account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            type="redeem",
            points=-200,
            source="reward_redemption",
        )
        db_session.add(pt)
        db_session.flush()
        assert pt.points == -200

    def test_history_ordering(self, db_session):
        account = "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF"
        from datetime import datetime, timezone, timedelta
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i, pts in enumerate([50, 100, -30]):
            pt = PointsTransaction(
                account_id=account,
                type="earn" if pts > 0 else "redeem",
                points=pts,
                source=f"event_{i}",
            )
            db_session.add(pt)
        db_session.flush()

        rows = db_session.execute(
            select(PointsTransaction)
            .where(PointsTransaction.account_id == account)
            .order_by(PointsTransaction.id)
        ).scalars().all()
        assert len(rows) == 3
        assert rows[0].points == 50
        assert rows[2].points == -30

    def test_net_balance_calculation(self, db_session):
        account = "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT"
        for pts in [200, 50, -75]:
            db_session.add(PointsTransaction(
                account_id=account,
                type="earn" if pts > 0 else "redeem",
                points=pts,
            ))
        db_session.flush()

        rows = db_session.execute(
            select(PointsTransaction).where(PointsTransaction.account_id == account)
        ).scalars().all()
        net = sum(r.points for r in rows)
        assert net == 175

    def test_filter_by_type(self, db_session):
        account = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        for t, pts in [("earn", 100), ("redeem", -50), ("adjust", 10)]:
            db_session.add(PointsTransaction(account_id=account, type=t, points=pts))
        db_session.flush()

        earns = db_session.execute(
            select(PointsTransaction).where(
                PointsTransaction.account_id == account,
                PointsTransaction.type == "earn",
            )
        ).scalars().all()
        assert len(earns) == 1
