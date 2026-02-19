"""
Neo4j connection manager.

Provides a connection wrapper for Neo4j database operations.
"""

import os
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from neo4j import GraphDatabase, Driver, Session
from dotenv import load_dotenv

load_dotenv()


class Neo4jConnection:
    """Connection manager for Neo4j database."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j URI (default: from NEO4J_URI env var)
            user: Neo4j username (default: from NEO4J_USER env var)
            password: Neo4j password (default: from NEO4J_PASSWORD env var)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "genshin_story_qa")

        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        """Get or create the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def verify_connectivity(self) -> bool:
        """Verify that the connection to Neo4j is working."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False

    @contextmanager
    def session(self, database: str = "neo4j"):
        """
        Context manager for Neo4j session.

        Args:
            database: Database name (default: "neo4j")

        Yields:
            Neo4j session object
        """
        session = self.driver.session(database=database)
        try:
            yield session
        finally:
            session.close()

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        database: str = "neo4j",
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            params: Query parameters
            database: Database name

        Returns:
            List of result records as dictionaries
        """
        with self.session(database=database) as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def execute_write(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        database: str = "neo4j",
    ) -> List[Dict[str, Any]]:
        """
        Execute a write transaction.

        Args:
            query: Cypher query string
            params: Query parameters
            database: Database name

        Returns:
            List of result records as dictionaries
        """
        with self.session(database=database) as session:
            result = session.execute_write(
                lambda tx: list(tx.run(query, params or {}))
            )
            return [dict(record) for record in result]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def get_connection() -> Neo4jConnection:
    """Factory function to get a Neo4j connection."""
    return Neo4jConnection()


if __name__ == "__main__":
    # Test connection
    conn = Neo4jConnection()
    if conn.verify_connectivity():
        print("Successfully connected to Neo4j!")

        # Test query
        result = conn.execute("RETURN 'Hello, Neo4j!' as message")
        print(f"Test query result: {result}")
    else:
        print("Failed to connect to Neo4j. Is the server running?")

    conn.close()
