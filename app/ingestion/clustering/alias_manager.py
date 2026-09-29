# clustering/alias_manager.py
# ============================================================
# Alias Manager for Production Clustering Pipeline
# Handles:
#   • Check current alias target
#   • Create temp index names
#   • Atomic alias swap (old -> new)
#   • Delete old index safely
# ============================================================

from datetime import datetime
from elasticsearch import Elasticsearch
from app.ingestion.clustering.config import CLUSTERS_ALIAS, ES_HOST
from app.app_logger import LoggerFactory
logger = LoggerFactory.get_logger("alias_manager")


class AliasManager:
    """
    Handles alias operations:
        - Identify which index alias currently points to
        - Generate temp index names
        - Atomically swap alias
        - Delete old index
    """

    def __init__(self, alias: str = CLUSTERS_ALIAS):
        self.alias = alias
        self.es = Elasticsearch(ES_HOST)

    # ------------------------------------------------------------
    # Utility: generate temp index name
    # ------------------------------------------------------------
    def make_temp_index_name(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{self.alias}_{ts}"

    # ------------------------------------------------------------
    # Check if alias exists and return index
    # ------------------------------------------------------------
    def get_current_active_index(self) -> str | None:
        """
        Returns the index name currently linked to the alias.
        Returns None if alias does not exist.
        """
        try:
            resp = self.es.indices.get_alias(name=self.alias)
        except Exception:
            return None

        # resp = { "index1": { "aliases": { "clusters_v2": {} } }, ... }
        for index_name in resp.keys():
            return index_name

        return None

    # ------------------------------------------------------------
    # Atomic alias swap
    # ------------------------------------------------------------
    def swap_alias(self, new_index: str):
        """
        Atomically remove alias from old index and assign it to new_index.
        Returns (old_index, new_index).
        """

        current = self.get_current_active_index()
        old_index = current

        actions = []

        if current:
            actions.append({
                "remove": {"index": current, "alias": self.alias}
            })

        actions.append({
            "add": {"index": new_index, "alias": self.alias}
        })

        logger.info(f"[ALIAS] Swapping alias: {self.alias}")
        logger.info(f"[ALIAS] Old → {old_index}, New → {new_index}")
        logger.info(f"[ALIAS] Actions = {actions}")

        # FIX — use keyword argument body=
        self.es.indices.update_aliases(body={"actions": actions})

        return old_index, new_index

    # ------------------------------------------------------------
    # Delete old index safely
    # ------------------------------------------------------------
    def delete_index(self, index_name: str):
        if not index_name:
            return

        logger.info(f"[ALIAS] Deleting old index: {index_name}")

        try:
            self.es.indices.delete(index=index_name, ignore=[404])
        except Exception as e:
            logger.error(f"[ALIAS] Failed to delete index {index_name}: {e}")

    # ------------------------------------------------------------
    # Full workflow: swap alias + delete old index
    # ------------------------------------------------------------
    def finalize_new_index(self, new_index: str, delete_old: bool = True):
        """
        1. Swap alias to new_index.
        2. Delete old index if delete_old = True.
        """
        old_index, _ = self.swap_alias(new_index)

        if delete_old and old_index:
            self.delete_index(old_index)

        logger.info(f"[ALIAS] Switchover complete: {self.alias} → {new_index}")
