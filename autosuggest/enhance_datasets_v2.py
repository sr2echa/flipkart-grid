import pandas as pd
import numpy as np
from faker import Faker
import random

class DatasetEnhancerV2:
    """
    Generates a more sophisticated and contextually rich dataset for the autosuggest system.
    """
    def __init__(self):
        self.fake = Faker('en_IN')

    def run_enhancement(self, output_user_queries_path, output_session_log_path):
        """Main function to generate and save enhanced datasets."""
        print("🚀 Starting V2 Dataset Enhancement...")
        
        # Generate User Queries
        user_queries = self.generate_user_queries(num_queries=15000)
        user_queries.to_csv(output_user_queries_path, index=False)
        print(f"✅ Generated and saved {len(user_queries)} enhanced user queries to {output_user_queries_path}")

        # Generate Session Logs
        session_log = self.generate_session_log(user_queries, num_sessions=20000)
        session_log.to_csv(output_session_log_path, index=False)
        print(f"✅ Generated and saved {len(session_log)} enhanced session log records to {output_session_log_path}")
        
        print("🎉 V2 Dataset Enhancement Complete!")

    def generate_user_queries(self, num_queries):
        """Generates a high-quality user queries dataset."""
        queries = []
        
        personas = {
            'tech_enthusiast': self._tech_enthusiast_queries,
            'fashion_lover': self._fashion_lover_queries,
            'budget_shopper': self._budget_shopper_queries,
            'sports_enthusiast': self._sports_enthusiast_queries,
            'luxury_shopper': self._luxury_shopper_queries,
            'home_maker': self._home_maker_queries,
        }
        
        for _ in range(num_queries):
            persona_id = random.choice(list(personas.keys()))
            query_generator = personas[persona_id]
            
            raw_query, corrected_query = query_generator()
            
            queries.append({
                'query_id': self.fake.uuid4(),
                'user_id': self.fake.uuid4(),
                'raw_query': raw_query,
                'corrected_query': corrected_query,
                'query_timestamp': self.fake.date_time_this_year(),
                'location': self.fake.city(),
                'persona_tag': persona_id,
            })
            
        df = pd.DataFrame(queries)
        df['frequency'] = df.groupby('corrected_query')['corrected_query'].transform('count')
        return df

    def generate_session_log(self, user_queries_df, num_sessions):
        """Generates a high-quality session log dataset."""
        sessions = []
        
        for _ in range(num_sessions):
            session_id = self.fake.uuid4()
            persona_tag = random.choice(user_queries_df['persona_tag'].unique())
            session_queries = user_queries_df[user_queries_df['persona_tag'] == persona_tag].sample(n=random.randint(2, 8))
            
            location = self.fake.city()
            event = random.choice(['Diwali', 'IPL', 'Christmas', 'Wedding Season', 'Summer Sale', 'None'])
            
            for _, row in session_queries.iterrows():
                sessions.append({
                    'session_id': session_id,
                    'timestamp': self.fake.date_time_this_year(),
                    'query': row['corrected_query'],
                    'clicked_product_id': self.fake.uuid4() if random.random() > 0.3 else None,
                    'purchased': True if random.random() > 0.8 else False,
                    'location': location,
                    'event': event,
                    'persona_tag': persona_tag
                })
        
        return pd.DataFrame(sessions)

    # --- Persona-specific Query Generators ---

    def _add_typo(self, query):
        if len(query) < 4 or random.random() > 0.4:
            return query, query
        
        pos = random.randint(1, len(query) - 2)
        typo_type = random.random()
        
        raw_query = ""
        if typo_type < 0.33: # substitution
            raw_query = query[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + query[pos+1:]
        elif typo_type < 0.66: # deletion
            raw_query = query[:pos] + query[pos+1:]
        else: # insertion
            raw_query = query[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + query[pos:]
            
        return raw_query, query

    def _tech_enthusiast_queries(self):
        templates = [
            f"{random.choice(['latest', 'new', 'gaming'])} {random.choice(['laptop', 'phone', 'motherboard'])}",
            f"{random.choice(['samsung', 'apple', 'oneplus', 'dell', 'nvidia'])} {random.choice(['rtx 4090', 's24 ultra', 'macbook pro', 'alienware'])}",
            f"best {random.choice(['cpu for gaming', 'mechanical keyboard', '4k monitor'])}",
            f"{random.choice(['wireless earbuds', 'noise cancelling headphones'])}",
        ]
        return self._add_typo(random.choice(templates))

    def _fashion_lover_queries(self):
        templates = [
            f"{random.choice(['nike', 'adidas', 'zara', 'h&m'])} {random.choice(['sneakers', 't-shirt', 'jeans', 'dress'])}",
            f"{random.choice(['summer', 'winter'])} fashion for {random.choice(['men', 'women'])}",
            f"aesthetic {random.choice(['tote bag', 'sunglasses'])}",
            f"trending {random.choice(['shoes', 'outfits'])} 2024",
        ]
        return self._add_typo(random.choice(templates))

    def _budget_shopper_queries(self):
        templates = [
            f"{random.choice(['phone', 'earphones', 'shoes'])} under {random.choice([500, 1000, 5000, 10000])}",
            f"cheap and best {random.choice(['t-shirts', 'kitchen items'])}",
            f"discounts on {random.choice(['laptops', 'tvs'])}",
            f"value for money {random.choice(['smartwatch', 'running shoes'])}",
        ]
        return self._add_typo(random.choice(templates))

    def _sports_enthusiast_queries(self):
        templates = [
            f"{random.choice(['nike', 'adidas', 'puma'])} {random.choice(['running shoes', 'football', 'cricket bat'])}",
            f"high protein {random.choice(['powder', 'bar'])}",
            f"{random.choice(['csk', 'mumbai indians', 'rcb'])} jersey",
            f"{random.choice(['home gym equipment', 'yoga mat', 'dumbbells'])}",
        ]
        return self._add_typo(random.choice(templates))

    def _luxury_shopper_queries(self):
        templates = [
            f"{random.choice(['gucci', 'louis vuitton', 'rolex'])} {random.choice(['bag', 'watch', 'belt'])}",
            f"premium {random.choice(['perfume for men', 'skincare'])}",
            f"latest {random.choice(['dyson airwrap', 'bose headphones'])}",
            f"{random.choice(['apple watch ultra', 'samsung fold 5'])}",
        ]
        return self._add_typo(random.choice(templates))

    def _home_maker_queries(self):
        templates = [
            f"best {random.choice(['mixer grinder', 'pressure cooker', 'air fryer'])}",
            f"{random.choice(['organic vegetables', 'atta', 'basmati rice'])}",
            f"non-stick {random.choice(['kadai', 'tawa'])}",
            f"home cleaning {random.choice(['liquids', 'mops', 'vacuum cleaner'])}",
        ]
        return self._add_typo(random.choice(templates))

if __name__ == "__main__":
    enhancer = DatasetEnhancerV2()
    enhancer.run_enhancement(
        output_user_queries_path='../dataset/user_queries_enhanced_v2.csv',
        output_session_log_path='../dataset/session_log_enhanced_v2.csv'
    )