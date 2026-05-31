# Databases & Data Engineering

Database systems and data engineering concepts for storing, querying, and managing data at scale.

## Topics Covered

### Database Design & Internals

- **[Database Design](database_design.md)** - Schema design, normalization, relationships, indexing strategies, and best practices
- **[Database Internals](database_internals.md)** - Pages & buffer pool, B-tree vs LSM storage engines, WAL durability, MVCC, isolation levels
- **[Indexing](indexing.md)** - B-tree/hash/GIN/BRIN indexes, clustered vs secondary, composite & covering indexes
- **[Query Optimization](query_optimization.md)** - Cost-based planner, EXPLAIN, join algorithms, statistics, SARGability
- **[Replication & Sharding](replication_sharding.md)** - Leader/follower & leaderless replication, sync vs async, partitioning, failover
- **[ACID vs BASE](acid_vs_base.md)** - Consistency models and trade-offs

### Relational Databases

- **[SQL](sql.md)** - SQL fundamentals, queries, joins, indexes, transactions
- **[PostgreSQL](postgres.md)** - Advanced PostgreSQL features, JSON support, performance tuning
- **[MySQL](mysql.md)** - InnoDB internals, storage engines, replication, and Postgres differences
- **[SQLite](sqlite.md)** - Lightweight embedded database for applications
- **[DuckDB](duckdb.md)** - Analytical database for data analysis and OLAP queries
- **[ClickHouse](clickhouse.md)** - Column-oriented OLAP database for real-time analytics at scale

### NoSQL Databases

- **[NoSQL](nosql.md)** - NoSQL databases overview, types, and use cases
- **[MongoDB](mongodb.md)** - Document-oriented NoSQL database with rich query language
- **[Cassandra](cassandra.md)** - Wide-column, leaderless distributed store with tunable consistency
- **[Elasticsearch](elasticsearch.md)** - Distributed full-text search & analytics on an inverted index
- **[Redis](redis.md)** - In-memory data store for caching, pub/sub, and real-time applications

### Message Queues & Event Streaming

- **[Apache Kafka](kafka.md)** - Distributed event streaming platform for high-throughput data pipelines
- **[Change Data Capture & Streaming](cdc_streaming.md)** - Log-based CDC, the outbox pattern, and stream processing

## Database Concepts

- **Data Modeling**: Schema design, normalization, relationships
- **Caching**: In-memory stores, cache invalidation strategies
- **Data Pipelines**: ETL, streaming, batch processing
- **Database Optimization**: Query optimization, indexing strategies

## Navigation

Use the menu to explore each topic in depth.
