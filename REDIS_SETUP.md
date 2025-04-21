# Redis Implementation for Word2Vec Visualization

This document describes how Redis is used in this project to store word embeddings and enable incremental visualization.

## Overview

The application now uses Redis as a persistent storage backend instead of in-memory storage. This allows:

1. Data persistence between server restarts
2. Incremental addition of words to the visualization
3. Improved scalability for larger word sets

## Redis Data Structure

The data is organized in Redis as follows:

- **Word Set**: `all_words` - A Redis Set containing all words added to the system
- **Word Embeddings**: `embedding:{word}` - Individual keys storing the pickled numpy arrays representing word embeddings

## Setup Instructions

### 1. Install Redis

#### Windows
- Download and install Redis for Windows from [https://github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)
- Or use WSL2 (Windows Subsystem for Linux) to run Redis

#### macOS
```bash
brew install redis
brew services start redis
```

#### Linux
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
```

### 2. Configure Environment

The application uses a `.env` file in the backend directory with the following variables:

```
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=
REDIS_DB=0
MODEL_NAME=bert-base-uncased
```

Customize these values for your environment.

### 3. Redis Security (Production)

For production deployments:

1. Set a strong password in Redis configuration:
   ```
   # In redis.conf
   requirepass your_strong_password
   ```

2. Update the `.env` file:
   ```
   REDIS_PASSWORD=your_strong_password
   ```

3. Consider using Redis Cloud or a managed Redis service for production.

## Redis Commands for Debugging

Useful Redis commands for debugging:

```bash
# Connect to Redis CLI
redis-cli

# List all keys
KEYS *

# Get all words in the set
SMEMBERS all_words

# Count all words
SCARD all_words

# Delete all data
FLUSHDB
```

## Data Persistence

Redis offers two persistence options:

1. **RDB (Redis Database)**: Point-in-time snapshots
2. **AOF (Append Only File)**: Logs every write operation

For this application, the default RDB configuration is sufficient, but you may want to adjust these settings for production use.

## Scaling Considerations

For larger word sets:

1. Consider increasing the `maxmemory` setting in `redis.conf`
2. Monitor memory usage with `INFO memory` command
3. Consider implementing a TTL (Time-To-Live) mechanism for less frequently used words

## Troubleshooting

If the application cannot connect to Redis:

1. Verify Redis is running: `redis-cli ping` (should return PONG)
2. Check connection settings in `.env`
3. Ensure your firewall allows connections to the Redis port
4. Check Redis logs: `/var/log/redis/redis-server.log` (Linux) 