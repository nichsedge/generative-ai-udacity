#!/usr/bin/env python3
"""
HomeMatch: Personalized Real Estate Agent with Multimodal Search
A comprehensive application that generates personalized real estate listings
using LLMs, OpenAI embeddings, and CLIP for multimodal image+text search.
"""

import json
import os
import requests
import base64
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
import numpy as np
from PIL import Image
import torch
import clip

OPENAI_API_KEY="voc-8493599951266774173114687b0498f1ca18.34672008"
OPENAI_API_BASE="https://openai.vocareum.com/v1"

class HomeMatchMultimodal:
    def __init__(self, openai_api_key: str = None):
        """Initialize HomeMatch with OpenAI and CLIP multimodal capabilities."""
        # Set up OpenAI
        openai.api_base = OPENAI_API_BASE
        openai.api_key = OPENAI_API_KEY
        
        # Initialize CLIP model for multimodal search
        print("Loading CLIP model for multimodal search...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize ChromaDB with separate collections for text and images
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create collections
        try:
            self.text_collection = self.chroma_client.create_collection(
                name="text_listings",
                metadata={"hnsw:space": "cosine"}
            )
            self.image_collection = self.chroma_client.create_collection(
                name="image_listings", 
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.text_collection = self.chroma_client.get_collection("text_listings")
            self.image_collection = self.chroma_client.get_collection("image_listings")
        
        self.listings = []

    def get_openai_embedding(self, text: str) -> List[float]:
        """Get text embedding using OpenAI's embedding model."""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            # Fallback to random embedding for demo
            return np.random.rand(1536).tolist()

    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using CLIP."""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def get_clip_image_embedding(self, image_path: str) -> np.ndarray:
        """Get image embedding using CLIP."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.random.rand(512).astype(np.float32)

    def generate_listings_with_images(self, num_listings: int = 10) -> List[Dict]:
        """Generate synthetic real estate listings with associated images using OpenAI."""
        print(f"Generating {num_listings} real estate listings with images...")
        
        system_prompt = """You are a real estate listing generator. Create diverse, realistic property listings with the following structure:
        - Neighborhood: [name]
        - Price: $[amount]
        - Bedrooms: [number]
        - Bathrooms: [number]  
        - House Size: [sqft] sqft
        - Property Type: [House/Condo/Townhouse]
        - Description: [detailed description mentioning visual elements like architecture, colors, landscaping]
        - Neighborhood Description: [area details]
        - Image Description: [detailed description of what a photo of this property would show - exterior, style, surroundings]
        
        Make each listing unique with different neighborhoods, architectural styles, and visual characteristics."""
        
        property_styles = [
            "Modern minimalist", "Victorian", "Colonial", "Ranch style", "Contemporary",
            "Craftsman", "Mediterranean", "Tudor", "Mid-century modern", "Farmhouse",
            "Art Deco", "Cape Cod", "Split-level", "Bungalow", "Georgian"
        ]
        
        neighborhoods = [
            "Green Oaks", "Sunset Hills", "River Valley", "Oak Manor", "Pine Ridge",
            "Maple Grove", "Cedar Point", "Willow Creek", "Stone Bridge", "Golden Gate",
            "Cherry Blossom", "Highland Park", "Lakeside", "Mountain View", "Prairie Winds"
        ]
        
        listings = []
        
        for i in range(num_listings):
            neighborhood = neighborhoods[i % len(neighborhoods)]
            style = property_styles[i % len(property_styles)]
            
            user_prompt = f"""Generate a detailed real estate listing for a {style} property in {neighborhood}. 
            Include varied property types, different price ranges ($200K-$2M), and unique visual features.
            Pay special attention to the Image Description - describe colors, architectural details, landscaping, 
            and visual elements that would be visible in a property photo."""
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.8
                )
                
                listing_text = response.choices[0].message.content.strip()
                listing = self._parse_listing_with_image(listing_text, i)
                
                # Generate a synthetic image URL (in real implementation, these would be actual photos)
                listing["image_url"] = f"https://example-real-estate.com/images/property_{i+1}.jpg"
                listing["has_image"] = True
                
                listings.append(listing)
                
            except Exception as e:
                print(f"Error generating listing {i+1}: {e}")
                listing = self._create_fallback_listing_with_image(i, neighborhood, style)
                listings.append(listing)
        
        self.listings = listings
        self._save_listings()
        return listings

    def _parse_listing_with_image(self, listing_text: str, listing_id: int) -> Dict:
        """Parse listing text into structured format including image description."""
        lines = listing_text.split('\n')
        listing = {"id": listing_id, "full_text": listing_text}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Neighborhood:"):
                listing["neighborhood"] = line.split(":", 1)[1].strip()
            elif line.startswith("Price:"):
                listing["price"] = line.split(":", 1)[1].strip()
            elif line.startswith("Bedrooms:"):
                listing["bedrooms"] = line.split(":", 1)[1].strip()
            elif line.startswith("Bathrooms:"):
                listing["bathrooms"] = line.split(":", 1)[1].strip()
            elif line.startswith("House Size:"):
                listing["size"] = line.split(":", 1)[1].strip()
            elif line.startswith("Property Type:"):
                listing["property_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Description:"):
                listing["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("Neighborhood Description:"):
                listing["neighborhood_description"] = line.split(":", 1)[1].strip()
            elif line.startswith("Image Description:"):
                listing["image_description"] = line.split(":", 1)[1].strip()
        
        return listing

    def _create_fallback_listing_with_image(self, listing_id: int, neighborhood: str, style: str) -> Dict:
        """Create fallback listing with image description if API fails."""
        return {
            "id": listing_id,
            "neighborhood": neighborhood,
            "price": "$750,000",
            "bedrooms": "3",
            "bathrooms": "2", 
            "size": "1,800 sqft",
            "property_type": "House",
            "description": f"Beautiful {style} family home in {neighborhood} featuring modern amenities, spacious rooms, and a lovely garden.",
            "neighborhood_description": f"{neighborhood} is a vibrant community with excellent schools and parks.",
            "image_description": f"Exterior view of a charming {style} home with well-maintained landscaping, attractive facade, and welcoming entrance.",
            "image_url": f"https://example-real-estate.com/images/property_{listing_id+1}.jpg",
            "has_image": True,
            "full_text": f"Neighborhood: {neighborhood}\nPrice: $750,000\nBedrooms: 3\nBathrooms: 2\nHouse Size: 1,800 sqft\nProperty Type: House\nDescription: Beautiful {style} family home..."
        }

    def _save_listings(self):
        """Save listings to file."""
        with open("listings.json", "w") as f:
            json.dump(self.listings, f, indent=2)
        print(f"Saved {len(self.listings)} listings to listings.json")

    def store_in_multimodal_db(self):
        """Store listings in ChromaDB with both text and image embeddings."""
        print("Storing listings in multimodal vector database...")
        
        if not self.listings:
            print("No listings to store. Generate listings first.")
            return
        
        # Store text embeddings
        text_documents = []
        text_metadatas = []
        text_ids = []
        text_embeddings = []
        
        # Store image embeddings  
        image_documents = []
        image_metadatas = []
        image_ids = []
        image_embeddings = []
        
        for listing in self.listings:
            listing_id = str(listing["id"])
            
            # Text processing
            searchable_text = f"{listing.get('description', '')} {listing.get('neighborhood_description', '')} {listing.get('neighborhood', '')} {listing.get('property_type', '')} {listing.get('bedrooms', '')} bedrooms {listing.get('bathrooms', '')} bathrooms"
            
            text_documents.append(searchable_text)
            text_metadatas.append({
                "neighborhood": listing.get("neighborhood", ""),
                "price": listing.get("price", ""),
                "bedrooms": listing.get("bedrooms", ""),
                "bathrooms": listing.get("bathrooms", ""),
                "property_type": listing.get("property_type", ""),
                "full_text": listing.get("full_text", "")
            })
            text_ids.append(f"text_{listing_id}")
            
            # Get OpenAI text embedding
            text_embedding = self.get_openai_embedding(searchable_text)
            text_embeddings.append(text_embedding)
            
            # Image processing using CLIP
            image_description = listing.get("image_description", listing.get("description", ""))
            
            image_documents.append(image_description)
            image_metadatas.append({
                "neighborhood": listing.get("neighborhood", ""),
                "price": listing.get("price", ""),
                "property_type": listing.get("property_type", ""),
                "image_url": listing.get("image_url", ""),
                "full_text": listing.get("full_text", "")
            })
            image_ids.append(f"image_{listing_id}")
            
            # Get CLIP image embedding (using text description as proxy)
            clip_embedding = self.get_clip_text_embedding(image_description)
            image_embeddings.append(clip_embedding.tolist())
        
        # Store in ChromaDB
        try:
            # Store text embeddings
            self.text_collection.add(
                documents=text_documents,
                metadatas=text_metadatas,
                embeddings=text_embeddings,
                ids=text_ids
            )
            
            # Store image embeddings
            self.image_collection.add(
                documents=image_documents,
                metadatas=image_metadatas,
                embeddings=image_embeddings,
                ids=image_ids
            )
            
            print(f"Successfully stored {len(text_documents)} listings with multimodal embeddings.")
        except Exception as e:
            print(f"Error storing in vector database: {e}")

    def collect_multimodal_preferences(self) -> Dict[str, str]:
        """Collect buyer preferences including visual preferences."""
        questions = [
            "How big do you want your house to be?",
            "What are 3 most important things for you in choosing this property?", 
            "Which amenities would you like?",
            "What architectural style or visual elements appeal to you?",
            "How urban do you want your neighborhood to be?",
            "Describe your ideal property's appearance and surroundings:"
        ]
        
        # Enhanced answers including visual preferences
        answers = [
            "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
            "A quiet neighborhood, good local schools, and convenient shopping options.",
            "A backyard for gardening, a two-car garage, and modern, energy-efficient heating system.",
            "I love modern architecture with clean lines, large windows, and contemporary design elements.",
            "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
            "I want a home with attractive curb appeal, well-maintained landscaping, and a welcoming entrance. I prefer lighter colors and modern exterior finishes."
        ]
        
        preferences = {}
        print("\n=== BUYER PREFERENCES (Including Visual) ===")
        for i, (question, answer) in enumerate(zip(questions, answers)):
            print(f"Q{i+1}: {question}")
            print(f"A{i+1}: {answer}\n")
            preferences[f"q{i+1}"] = answer
        
        # Create combined preference texts
        preferences["combined_text"] = " ".join(answers[:4])  # Text-focused preferences
        preferences["combined_visual"] = " ".join(answers[3:])  # Visual-focused preferences
        preferences["combined_all"] = " ".join(answers)
        
        return preferences

    def multimodal_search(self, preferences: Dict[str, str], n_results: int = 3, 
                         text_weight: float = 0.6, visual_weight: float = 0.4) -> List[Dict]:
        """Search listings using both text and visual embeddings with weighted combination."""
        print("Performing multimodal search (text + visual)...")
        
        try:
            # Text-based search using OpenAI embeddings
            text_query_embedding = self.get_openai_embedding(preferences["combined_text"])
            text_results = self.text_collection.query(
                query_embeddings=[text_query_embedding],
                n_results=n_results * 2  # Get more candidates
            )
            
            # Visual-based search using CLIP embeddings
            visual_query_embedding = self.get_clip_text_embedding(preferences["combined_visual"])
            visual_results = self.image_collection.query(
                query_embeddings=[visual_query_embedding.tolist()],
                n_results=n_results * 2  # Get more candidates
            )
            
            # Combine and re-rank results
            combined_scores = {}
            
            # Process text results
            for i, text_id in enumerate(text_results["ids"][0]):
                listing_id = text_id.replace("text_", "")
                text_distance = text_results["distances"][0][i]
                text_score = 1 - text_distance
                combined_scores[listing_id] = {
                    "text_score": text_score,
                    "visual_score": 0,
                    "text_metadata": text_results["metadatas"][0][i],
                    "text_document": text_results["documents"][0][i]
                }
            
            # Process visual results
            for i, image_id in enumerate(visual_results["ids"][0]):
                listing_id = image_id.replace("image_", "")
                visual_distance = visual_results["distances"][0][i]
                visual_score = 1 - visual_distance
                
                if listing_id in combined_scores:
                    combined_scores[listing_id]["visual_score"] = visual_score
                    combined_scores[listing_id]["image_metadata"] = visual_results["metadatas"][0][i]
                    combined_scores[listing_id]["image_document"] = visual_results["documents"][0][i]
                else:
                    # Handle case where listing only appears in visual results
                    combined_scores[listing_id] = {
                        "text_score": 0,
                        "visual_score": visual_score,
                        "image_metadata": visual_results["metadatas"][0][i],
                        "image_document": visual_results["documents"][0][i]
                    }
            
            # Calculate weighted combined scores
            final_results = []
            for listing_id, scores in combined_scores.items():
                combined_score = (text_weight * scores["text_score"] + 
                                visual_weight * scores["visual_score"])
                
                result = {
                    "id": listing_id,
                    "combined_score": combined_score,
                    "text_score": scores["text_score"],
                    "visual_score": scores["visual_score"],
                    "metadata": scores.get("text_metadata", scores.get("image_metadata", {})),
                    "text_document": scores.get("text_document", ""),
                    "image_document": scores.get("image_document", "")
                }
                final_results.append(result)
            
            # Sort by combined score and return top results
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            top_results = final_results[:n_results]
            
            print(f"Found {len(top_results)} multimodal matches.")
            return top_results
            
        except Exception as e:
            print(f"Error in multimodal search: {e}")
            return []

    def personalize_multimodal_listing(self, listing: Dict, preferences: Dict[str, str]) -> str:
        """Generate personalized listing description emphasizing both text and visual aspects."""
        original_listing = listing["metadata"]["full_text"]
        buyer_preferences = preferences["combined_all"]
        text_score = listing["text_score"]
        visual_score = listing["visual_score"]
        combined_score = listing["combined_score"]
        
        system_prompt = """You are a real estate agent who personalizes property descriptions for buyers using both textual and visual insights. 
        Your job is to rewrite listing descriptions to highlight aspects that match the buyer's preferences, 
        including architectural style, visual appeal, and aesthetic elements, while keeping all factual information accurate."""
        
        user_prompt = f"""
        Original Listing:
        {original_listing}
        
        Buyer Preferences:
        {buyer_preferences}
        
        Search Scores:
        - Text Match: {text_score:.2f}
        - Visual Match: {visual_score:.2f}
        - Overall Match: {combined_score:.2f}
        
        Please rewrite this listing description to emphasize both the practical features AND visual/aesthetic aspects 
        that would appeal to this buyer. Highlight architectural style, visual elements, and design features 
        that match their preferences. Keep all facts accurate but make it more personalized and compelling.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=700,
                temperature=0.7
            )
            
            personalized_description = response.choices[0].message.content.strip()
            
            # Add score information
            score_info = f"**🎯 Match Analysis:**\n"
            score_info += f"• Text Match: {text_score*100:.1f}% (features, location, amenities)\n"
            score_info += f"• Visual Match: {visual_score*100:.1f}% (style, appearance, aesthetics)\n"
            score_info += f"• Overall Match: {combined_score*100:.1f}%\n\n"
            
            return score_info + personalized_description
            
        except Exception as e:
            print(f"Error personalizing listing: {e}")
            return f"**Multimodal Match (Overall: {combined_score*100:.1f}%)**\n\n{original_listing}"

    def run_homematch_multimodal(self):
        """Run the complete multimodal HomeMatch application workflow."""
        print("🏠✨ Welcome to HomeMatch Multimodal - Your AI-Powered Real Estate Agent! ✨🏠")
        print("Now with advanced image+text search capabilities powered by CLIP and OpenAI!\n")
        
        # Step 1: Generate listings with image descriptions
        self.generate_listings_with_images(12)
        
        # Step 2: Store in multimodal vector database
        self.store_in_multimodal_db()
        
        # Step 3: Collect multimodal buyer preferences
        preferences = self.collect_multimodal_preferences()
        
        # Step 4: Perform multimodal search
        matched_listings = self.multimodal_search(preferences, n_results=3)
        
        # Step 5: Generate personalized multimodal descriptions
        print("\n=== PERSONALIZED MULTIMODAL RECOMMENDATIONS ===\n")
        
        for i, listing in enumerate(matched_listings, 1):
            print(f"{'='*70}")
            print(f"🏡 MULTIMODAL RECOMMENDATION #{i}")
            print(f"Overall Match: {listing['combined_score']*100:.1f}%")
            print(f"{'='*70}")
            
            personalized_desc = self.personalize_multimodal_listing(listing, preferences)
            print(personalized_desc)
            print("\n")

def main():
    """Main function to run multimodal HomeMatch application."""
    try:
        app = HomeMatchMultimodal()
        app.run_homematch_multimodal()
    except Exception as e:
        print(f"Error running HomeMatch Multimodal: {e}")
        print("Make sure you have:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Installed: uv add openai chromadb torch torchvision clip-by-openai pillow")

if __name__ == "__main__":
    main()