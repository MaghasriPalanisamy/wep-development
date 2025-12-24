import pandas as pd
import os

# Create datasets folder if it doesn't exist
os.makedirs('datasets', exist_ok=True)

# Sample data for Amazon
amazon_data = {
    'name': [
        'Apple iPhone 15 Pro',
        'Samsung Galaxy S24 Ultra',
        'Sony WH-1000XM5 Headphones',
        'Nike Air Max 270',
        'Dell XPS 15 Laptop',
        'Kindle Paperwhite',
        'Apple Watch Series 9',
        'Bose QuietComfort 45',
        'Logitech MX Master 3S',
        'Canon EOS R50 Camera'
    ],
    'category': [
        'smartphone',
        'smartphone',
        'headphones',
        'shoes',
        'laptop',
        'ebook reader',
        'smartwatch',
        'headphones',
        'mouse',
        'camera'
    ],
    'price': [999.99, 1299.99, 399.99, 150.00, 1499.99, 139.99, 399.00, 329.00, 99.99, 679.00],
    'brand': [
        'Apple',
        'Samsung',
        'Sony',
        'Nike',
        'Dell',
        'Amazon',
        'Apple',
        'Bose',
        'Logitech',
        'Canon'
    ],
    'rating': [4.8, 4.7, 4.9, 4.6, 4.5, 4.8, 4.7, 4.8, 4.6, 4.4],
    'stock': ['In Stock'] * 10,
    'discount_percent': [5, 10, 15, 20, 5, 10, 8, 12, 15, 7],
    'description': [
        'Latest iPhone with A17 Pro chip',
        'Samsung flagship smartphone with S-Pen',
        'Premium noise cancelling wireless headphones',
        'Comfortable running shoes with Air cushioning',
        'Powerful laptop with OLED display',
        'Waterproof e-reader with adjustable light',
        'Advanced smartwatch with health monitoring',
        'Noise cancelling headphones with premium sound',
        'Wireless mouse with ergonomic design',
        'Mirrorless camera for photography enthusiasts'
    ],
    'model_id': [
        'IP15PRO256',
        'SGS24U512',
        'WH1000XM5',
        'AIRMAX270',
        'XPS159530',
        'KINDLEPW11',
        'AWS945MM',
        'BOSEQC45',
        'MXM3S',
        'EOSR50KIT'
    ],
    'url': [
        'https://www.amazon.com/dp/B0CHX1W1ZY',
        'https://www.amazon.com/dp/B0CM59T1SX',
        'https://www.amazon.com/dp/B09XS7JWHH',
        'https://www.amazon.com/dp/B07B3QGW8N',
        'https://www.amazon.com/dp/B0CJHQDZ2P',
        'https://www.amazon.com/dp/B09SWDBZQ4',
        'https://www.amazon.com/dp/B0CHWJQ85L',
        'https://www.amazon.com/dp/B098FKXT8L',
        'https://www.amazon.com/dp/B09HMK8M2P',
        'https://www.amazon.com/dp/B0BV8MPP4Q'
    ]
}

# Sample data for Flipkart
flipkart_data = {
    'name': [
        'OnePlus 12',
        'MacBook Air M2',
        'JBL Flip 6 Speaker',
        'Adidas Ultraboost 22',
        'LG OLED C3 TV',
        'GoPro HERO12',
        'Microsoft Surface Pro 9',
        'Fitbit Charge 6',
        'Philips Sonicare Toothbrush',
        'Instant Pot Duo'
    ],
    'category': [
        'smartphone',
        'laptop',
        'speaker',
        'shoes',
        'television',
        'camera',
        'tablet',
        'fitness tracker',
        'personal care',
        'kitchen appliance'
    ],
    'price': [849.99, 1099.99, 129.99, 180.00, 1499.00, 399.99, 999.99, 159.99, 89.99, 79.99],
    'brand': [
        'OnePlus',
        'Apple',
        'JBL',
        'Adidas',
        'LG',
        'GoPro',
        'Microsoft',
        'Fitbit',
        'Philips',
        'Instant Pot'
    ],
    'rating': [4.6, 4.8, 4.7, 4.5, 4.9, 4.6, 4.5, 4.4, 4.7, 4.8],
    'stock': ['In Stock'] * 10,
    'discount_percent': [12, 8, 20, 15, 10, 5, 12, 18, 10, 25],
    'description': [
        'Flagship killer smartphone',
        'Lightweight laptop with Apple Silicon',
        'Portable Bluetooth speaker',
        'Running shoes with Boost technology',
        '4K OLED smart TV',
        'Action camera for adventures',
        '2-in-1 laptop and tablet',
        'Advanced fitness and health tracker',
        'Electric toothbrush with smart features',
        'Multi-functional pressure cooker'
    ],
    'model_id': [
        'OP12PRO',
        'MBAM2',
        'JBLFLIP6',
        'UBOOST22',
        'OLED55C3',
        'HERO12',
        'SPRO9',
        'FITCHG6',
        'SONIC9900',
        'IPDUO7'
    ],
    'url': [
        'https://www.flipkart.com/oneplus-12',
        'https://www.flipkart.com/macbook-air-m2',
        'https://www.flipkart.com/jbl-flip-6',
        'https://www.flipkart.com/adidas-ultraboost',
        'https://www.flipkart.com/lg-oled-c3',
        'https://www.flipkart.com/gopro-hero12',
        'https://www.flipkart.com/surface-pro-9',
        'https://www.flipkart.com/fitbit-charge-6',
        'https://www.flipkart.com/philips-sonicare',
        'https://www.flipkart.com/instant-pot-duo'
    ]
}

# Create DataFrames
df_amazon = pd.DataFrame(amazon_data)
df_flipkart = pd.DataFrame(flipkart_data)

# Save to CSV
df_amazon.to_csv('datasets/amazon.csv', index=False)
df_flipkart.to_csv('datasets/flipkart.csv', index=False)

print("✅ Sample data created successfully!")
print(f"📊 Amazon: {len(df_amazon)} products")
print(f"📊 Flipkart: {len(df_flipkart)} products")
print(f"💾 Files saved in 'datasets/' folder")