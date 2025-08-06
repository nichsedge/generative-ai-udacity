import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)
embeddings = OpenAIEmbeddings(
    base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY")
)


def generate_listings(num_listings=10):
    """Generates real estate listings using an LLM."""
    template = """
    Generate a realistic real estate listing with the following details:
    Neighborhood: {neighborhood}
    Price: {price}
    Bedrooms: {bedrooms}
    Bathrooms: {bathrooms}
    House Size: {house_size} sqft
    Description: {description}
    Neighborhood Description: {neighborhood_description}

    Ensure the description is engaging and highlights unique features.
    """

    prompt = PromptTemplate(
        input_variables=[
            "neighborhood",
            "price",
            "bedrooms",
            "bathrooms",
            "house_size",
            "description",
            "neighborhood_description",
        ],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    listings = []
    listing_details = [
        {
            "neighborhood": "Green Oaks",
            "price": "$800,000",
            "bedrooms": "3",
            "bathrooms": "2",
            "house_size": "2,000",
            "description": "Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.",
            "neighborhood_description": "Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.",
        },
        {
            "neighborhood": "City Center",
            "price": "$1,200,000",
            "bedrooms": "2",
            "bathrooms": "2.5",
            "house_size": "1,500",
            "description": "Luxurious downtown condo with breathtaking city views. This modern 2-bedroom, 2.5-bathroom unit features high ceilings, a gourmet kitchen with state-of-the-art appliances, and a private balcony. Enjoy urban living at its finest with direct access to the city's best restaurants and entertainment.",
            "neighborhood_description": "The City Center is a vibrant, bustling area known for its nightlife, cultural attractions, and diverse culinary scene. Public transportation is abundant, and most amenities are within walking distance.",
        },
        {
            "neighborhood": "Lakeview",
            "price": "$650,000",
            "bedrooms": "4",
            "bathrooms": "3",
            "house_size": "2,800",
            "description": "Spacious family home with stunning lake views. This 4-bedroom, 3-bathroom property offers ample living space, a large backyard perfect for entertaining, and direct access to Lakeview Park. Ideal for families seeking tranquility and outdoor activities.",
            "neighborhood_description": "Lakeview is a serene residential area popular with families. It offers excellent schools, beautiful parks, and a strong sense of community. The lake provides opportunities for boating, fishing, and scenic walks.",
        },
        {
            "neighborhood": "Historic District",
            "price": "$950,000",
            "bedrooms": "3",
            "bathrooms": "2",
            "house_size": "1,800",
            "description": "Charming historic home meticulously restored to its original grandeur. This 3-bedroom, 2-bathroom house boasts original hardwood floors, intricate moldings, and a cozy fireplace. Located on a tree-lined street, it offers a blend of old-world charm and modern conveniences.",
            "neighborhood_description": "The Historic District is renowned for its beautiful architecture, cobblestone streets, and rich history. It's a quiet, picturesque neighborhood with local boutiques, art galleries, and cafes.",
        },
        {
            "neighborhood": "Tech Hub",
            "price": "$750,000",
            "bedrooms": "2",
            "bathrooms": "2",
            "house_size": "1,200",
            "description": "Modern loft in the heart of the Tech Hub. This sleek 2-bedroom, 2-bathroom unit features open-concept living, smart home technology, and a communal workspace. Perfect for tech professionals seeking a dynamic and convenient lifestyle.",
            "neighborhood_description": "The Tech Hub is a rapidly developing area, home to numerous tech companies, co-working spaces, and innovative startups. It offers a lively atmosphere with trendy cafes, modern eateries, and excellent public transport links.",
        },
        {
            "neighborhood": "Riverside",
            "price": "$550,000",
            "bedrooms": "3",
            "bathrooms": "2",
            "house_size": "1,900",
            "description": "Cozy riverside bungalow with direct access to walking trails. This 3-bedroom, 2-bathroom home offers a peaceful retreat with a large deck overlooking the river. Ideal for nature lovers and those seeking a quiet escape.",
            "neighborhood_description": "Riverside is a tranquil community known for its scenic riverfront, lush green spaces, and extensive walking and biking trails. It's a peaceful neighborhood with a strong sense of community and easy access to nature.",
        },
        {
            "neighborhood": "Artisan Village",
            "price": "$700,000",
            "bedrooms": "3",
            "bathrooms": "2.5",
            "house_size": "2,100",
            "description": "Unique artisan home with custom-built features and a spacious studio. This 3-bedroom, 2.5-bathroom property is perfect for creatives, offering a dedicated workspace, natural light, and a vibrant garden. Located in a community known for its artistic flair.",
            "neighborhood_description": "Artisan Village is a bohemian neighborhood celebrated for its vibrant arts scene, independent boutiques, and diverse culinary offerings. It's a walkable community with a lively atmosphere and frequent art events.",
        },
        {
            "neighborhood": "Family Meadows",
            "price": "$600,000",
            "bedrooms": "4",
            "bathrooms": "2.5",
            "house_size": "2,500",
            "description": "Traditional family home in a quiet cul-de-sac. This 4-bedroom, 2.5-bathroom house features a large fenced backyard, a two-car garage, and spacious living areas. Close to top-rated schools and community parks, it's ideal for growing families.",
            "neighborhood_description": "Family Meadows is a peaceful, family-oriented neighborhood with excellent schools, safe streets, and numerous parks. It's a suburban haven with a strong community spirit and convenient access to shopping centers.",
        },
        {
            "neighborhood": "University Heights",
            "price": "$450,000",
            "bedrooms": "2",
            "bathrooms": "1",
            "house_size": "1,000",
            "description": "Compact and convenient apartment near the university campus. This 2-bedroom, 1-bathroom unit is perfect for students or faculty, offering easy access to campus facilities, public transport, and local cafes.",
            "neighborhood_description": "University Heights is a lively, youthful neighborhood with a strong academic vibe. It offers a variety of affordable eateries, bookstores, and cultural events, all within walking distance of the university.",
        },
        {
            "neighborhood": "Coastal Breeze",
            "price": "$1,500,000",
            "bedrooms": "3",
            "bathrooms": "3",
            "house_size": "2,200",
            "description": "Stunning beachfront property with panoramic ocean views. This 3-bedroom, 3-bathroom home features direct beach access, a large patio for entertaining, and luxurious finishes throughout. Experience coastal living at its finest.",
            "neighborhood_description": "Coastal Breeze is an exclusive seaside community known for its pristine beaches, upscale dining, and recreational activities like surfing and sailing. It's a serene and picturesque neighborhood offering a high-quality coastal lifestyle.",
        },
    ]

    for i, details in enumerate(listing_details):
        print(f"Generating listing {i + 1}/{num_listings}...")
        response = chain.run(details)
        listings.append(response)

    return listings


def save_listings(listings, filename="listings.txt"):
    """Saves generated listings to a file."""
    with open(filename, "w") as f:
        for i, listing in enumerate(listings):
            f.write(f"""--- Listing {i + 1} ---""")
            f.write(listing)
            f.write("""\n""")


if __name__ == "__main__":
    print("Generating real estate listings...")
    generated_listings = generate_listings(num_listings=10)
    save_listings(generated_listings, "listings.txt")
    print(f"Generated {len(generated_listings)} listings and saved to listings.txt")

    # Step 3: Storing Listings in a Vector Database
    print("Storing listings in ChromaDB...")
    # For simplicity, we'll use the generated listings directly. In a real app, you might load them from the file.
    documents = generated_listings

    # Initialize ChromaDB with the listings and embeddings
    # This will create a persistent ChromaDB instance in the './chroma_db' directory
    vectorstore = Chroma.from_texts(
        texts=documents, embedding=embeddings, persist_directory="./chroma_db"
    )
    vectorstore.persist()
    print("Listings stored in ChromaDB.")

    # Step 4 & 5: Building the User Preference Interface and Searching
    print(
        print("""
Collecting buyer preferences and searching for matches.""")
    )
    questions = [
        "How big do you want your house to be?",
        "What are 3 most important things for you in choosing this property?",
        "Which amenities would you like?",
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
    ]
    answers = [
        "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
        "A quiet neighborhood, good local schools, and convenient shopping options.",
        "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
        "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
        "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
    ]

    buyer_preferences = " ".join([f"Q: {q} A: {a}" for q, a in zip(questions, answers)])
    print(f"""Buyer Preferences:
{buyer_preferences}""")

    # Perform semantic search
    print("""
Performing semantic search...""")
    # Retrieve top 3 most relevant listings
    retrieved_docs = vectorstore.similarity_search(buyer_preferences, k=3)
    print(f"Found {len(retrieved_docs)} relevant listings.")

    # Step 6: Personalizing Listing Descriptions
    print("""
Personalizing listing descriptions...""")
    personalized_listings = []
    personalization_template = """
    Based on the buyer's preferences: "{preferences}",
    rewrite the following real estate listing to highlight aspects most relevant to the buyer.
    Ensure factual information remains unchanged.

    Original Listing:
    {listing_description}

    Personalized Listing:
    """
    personalization_prompt = PromptTemplate(
        input_variables=["preferences", "listing_description"],
        template=personalization_template,
    )
    personalization_chain = LLMChain(llm=llm, prompt=personalization_prompt)

    for i, doc in enumerate(retrieved_docs):
        print(f"""
--- Personalizing Listing {i + 1} ---""")
        original_listing = doc.page_content
        personalized_description = personalization_chain.run(
            preferences=buyer_preferences, listing_description=original_listing
        )
        personalized_listings.append(personalized_description)
        print(personalized_description)