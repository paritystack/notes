# GraphQL

## Overview

GraphQL is a query language for APIs. Request exactly what data you need, no more, no less.

## Key Differences from REST

| Aspect | REST | GraphQL |
|--------|------|---------|
| **Endpoints** | Multiple (/users, /posts, /comments) | Single (/graphql) |
| **Data** | Fixed shape | Client specifies shape |
| **Over-fetching** | Get extra fields | Only requested fields |
| **Under-fetching** | Need multiple requests | Single request |

## Schema

Define types and their relationships:

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  posts: [Post!]!
  age: Int
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  createdAt: String!
}

type Query {
  user(id: ID!): User
  users: [User!]!
  post(id: ID!): Post
}

type Mutation {
  createUser(name: String!, email: String!): User!
  updateUser(id: ID!, name: String): User
  deleteUser(id: ID!): Boolean!
}
```

## Queries

Request exactly what you need:

```graphql
# Simple query
query {
  user(id: "1") {
    name
    email
  }
}

# Nested query
query {
  user(id: "1") {
    name
    posts {
      title
      createdAt
    }
  }
}

# Multiple queries
query {
  user1: user(id: "1") {
    name
  }
  user2: user(id: "2") {
    name
  }
}

# With variables
query GetUser($userId: ID!) {
  user(id: $userId) {
    name
    email
    posts {
      title
    }
  }
}
```

## Mutations

Modify data:

```graphql
mutation CreateUser($name: String!, $email: String!) {
  createUser(name: $name, email: $email) {
    id
    name
    email
  }
}

mutation UpdateUser($id: ID!, $name: String) {
  updateUser(id: $id, name: $name) {
    id
    name
  }
}
```

## Resolvers

Implement schema with resolvers:

```javascript
const resolvers = {
  Query: {
    user: (parent, args) => {
      return db.users.find(u => u.id === args.id);
    },
    users: () => {
      return db.users;
    }
  },

  Mutation: {
    createUser: (parent, args) => {
      const user = { id: uuidv4(), ...args };
      db.users.push(user);
      return user;
    }
  },

  User: {
    posts: (parent) => {
      return db.posts.filter(p => p.authorId === parent.id);
    }
  }
};
```

## Apollo Server (Node.js)

```javascript
const { ApolloServer, gql } = require('apollo-server');

const typeDefs = gql`
  type Query {
    hello: String
    user(id: ID!): User
  }

  type User {
    id: ID!
    name: String!
  }
`;

const resolvers = {
  Query: {
    hello: () => 'Hello world!',
    user: (_, args) => ({ id: args.id, name: 'John' })
  }
};

const server = new ApolloServer({
  typeDefs,
  resolvers
});

server.listen();
```

## Advantages

✅ Request only needed data (no over-fetching)
✅ Single request for related data (no under-fetching)
✅ Strong typing with schema
✅ Introspection (explore API automatically)
✅ Development tools (GraphQL Explorer)

## Disadvantages

❌ More complex than REST
❌ Query complexity attacks
❌ Caching is harder
❌ Monitoring harder
❌ Learning curve

## Best Practices

1. **Limit query depth** (prevent abuse)
2. **Implement timeout** on queries
3. **Use pagination** for large result sets
4. **Combine with REST** if needed
5. **Monitor query performance**

## Pagination

```graphql
query {
  users(first: 10, after: "cursor123") {
    edges {
      node {
        id
        name
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

## ELI10

GraphQL is like ordering food:
- **REST**: Get whole menu as-is
- **GraphQL**: Ask for exactly what you want

"I'll take pasta with sauce on the side, hold the onions"

## Further Resources

- [GraphQL Official](https://graphql.org/)
- [How to GraphQL](https://www.howtographql.com/)
- [Apollo Documentation](https://www.apollographql.com/docs/)
