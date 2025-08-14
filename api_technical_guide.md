# API Technical Documentation

## Overview
This document provides technical documentation for the Sample Company API system.

## Authentication
All API requests require authentication using Bearer tokens.
Include the token in the Authorization header:
```
Authorization: Bearer <your-token>
```

## Base URL
```
https://api.samplecompany.com/v1
```

## Endpoints

### Users
- `GET /users` - List all users
- `POST /users` - Create a new user
- `GET /users/{id}` - Get user by ID
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

### Products
- `GET /products` - List all products
- `POST /products` - Create a new product
- `GET /products/{id}` - Get product by ID
- `PUT /products/{id}` - Update product
- `DELETE /products/{id}` - Delete product

## Response Format
All API responses follow this format:
```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully"
}
```

## Error Handling
HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting
API requests are limited to 1000 requests per hour per API key.

## SDK Examples

### Python
```python
import requests

headers = {
    'Authorization': 'Bearer your-token',
    'Content-Type': 'application/json'
}

response = requests.get('https://api.samplecompany.com/v1/users', headers=headers)
users = response.json()
```

### JavaScript
```javascript
const response = await fetch('https://api.samplecompany.com/v1/users', {
    headers: {
        'Authorization': 'Bearer your-token',
        'Content-Type': 'application/json'
    }
});
const users = await response.json();
```

## Testing
Use the provided Postman collection for testing API endpoints.
All endpoints include example requests and responses.

## Support
For technical support, contact: tech-support@samplecompany.com
