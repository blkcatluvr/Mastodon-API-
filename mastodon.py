import requests
import json
import html2text
import csv
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline

access_token = 'VaErCz1zTCVXYhciajC4mEVGXHE2JsHQJFN0eywDuMY'
base_url = 'https://mastodon.social/api/v1/timelines/tag/freepalestine'
classifier = pipeline("text-classification", model="unitary/toxic-bert")


headers = {
    'Authorization': f'Bearer {access_token}'
}

def get_hashtag_posts(limit=40, max_pages=100, json_filename='mastodon_posts.json'):
    all_statuses = []
    url = base_url  
    page_counter = 0 

    with open(json_filename, mode='w', encoding='utf-8') as json_file:
        while url and page_counter < max_pages:
            response = requests.get(url, headers=headers, params={'limit': limit})

            if response.status_code == 200:
                search_results = response.json()

                if not search_results:
                    print("No more results.")
                    break  

                all_statuses.extend(search_results)  

                # Convert the data into JSON format and save
                for status in search_results:
                    # Convert HTML content to plain text
                    plain_text_content = html2text.html2text(status.get('content', ''))
                    
                    account = status.get('account', {})
                    author = f"@{account.get('username')}"

                    post_data = {
                        'post_id': status.get('id'),
                        'content': plain_text_content,
                        'created_at': status.get('created_at'),
                        'author': author,
                        'replies_count': status.get('replies_count', 0),
                        'in_reply_to_id': status.get('in_reply_to_id', None),  # Capture replies
                        'favourites_count': status.get('favourites_count', 0)
                    }

                    json.dump(post_data, json_file)
                    json_file.write('\n')  

                # Check if there's a "next" link in the headers for pagination
                if 'next' in response.links:
                    url = response.links['next']['url']  # Update the URL to the next page
                    page_counter += 1  
                else:
                    print("No next page.")
                    break  
            else:
                print(f"Error: {response.status_code}")
                break  

    print(f"Total posts fetched: {len(all_statuses)}")
    return all_statuses


def get_user_data(seed_users, max_followers=1000, json_filename='mastodon_users.json'):
    all_users = []
    base_user_url = 'https://mastodon.social/api/v1/accounts/'
    
    with open(json_filename, mode='w', encoding='utf-8') as json_file:
        for user in seed_users:
            print(f"Fetching user data for: {user}")
            response = requests.get(f"{base_user_url}search?q={user}", headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()

                if not user_data:
                    print(f"No data found for user: {user}")
                    continue

                user_id = user_data[0].get('id')  
                followers_url = f"{base_user_url}{user_id}/followers"
                followers_data = requests.get(followers_url, headers=headers, params={'limit': max_followers}).json()

                user_info = {
                    'user_id': user_id,
                    'username': user,
                    'followers': [f"@{follower.get('username')}" for follower in followers_data]
                }

                json.dump(user_info, json_file)
                json_file.write('\n')

                all_users.append(user_info)
            else:
                print(f"Error: {response.status_code} for user: {user}")

    return all_users

def get_top_users(csv_filename='mastodon_posts.csv', top_n=5):
    user_post_count = defaultdict(int)
    user_replies_count = defaultdict(int)
    user_likes_count = defaultdict(int)

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            author = row['author']
            user_post_count[author] += 1  # Count posts
            user_replies_count[author] += int(row.get('replies_count', 0))  # Count replies
            user_likes_count[author] += int(row.get('favourites_count', 0))  # Count likes 

    top_active_users = sorted(user_post_count, key=user_post_count.get, reverse=True)[:top_n]
    top_reply_users = sorted(user_replies_count, key=user_replies_count.get, reverse=True)[:top_n]
    top_like_users = sorted(user_likes_count, key=user_likes_count.get, reverse=True)[:top_n]

    top_users = set(top_active_users + top_reply_users + top_like_users)

    return top_users

def build_information_diffusion_network(json_filename='mastodon_posts.json', max_posts=600):
    G = nx.DiGraph() 
    post_counter = 0  

    with open(json_filename, 'r', encoding='utf-8') as f:
        for line in f:
            post = json.loads(line)
            post_id = post['post_id']
            G.add_node(post_id)  # Add each post as a node

            # Add edge if this post is a reply to another post
            if post['in_reply_to_id']:
                G.add_edge(post['in_reply_to_id'], post_id)  # Directed edge from original post to reply

            post_counter += 1  
            if post_counter >= max_posts:  
                break

    return G
def build_friendship_network(json_filename='mastodon_users.json'):
    G = nx.Graph() 
    
    with open(json_filename, 'r', encoding='utf-8') as f:
        for line in f:
            user_data = json.loads(line)
            username = user_data['username']
            G.add_node(username)  # Add the user as a node

            # Add edges between user and their followers
            for follower in user_data['followers']:
                G.add_edge(username, follower)
    
    return G

def extract_post_content(infoGraph, json_filename='mastodon_posts.json'):
    post_contents = {}
    with open(json_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if json.loads(line)['post_id'] in infoGraph.nodes():
                post = json.loads(line)
                post_id = post['post_id']
                post_contents[post_id] = post['content']
    
    return post_contents

def classify_toxicity(post_contents):
    print("Gathering Toxic")
    max_length=512
    classified_results = {}
    for post_id, content in post_contents.items():
        #Toxicity classification
        result = classifier(content[:max_length])
        label = result[0]['label']  # Get the toxicity classification
        classified_results[post_id] = label

    with open('classified_results.json', 'w', encoding='utf-8') as outfile:
        json.dump(classified_results, outfile, indent=4)
    
    return classified_results


def classify_nodes_in_network(graph, results_filename='classified_results.json'):
    with open(results_filename, 'r', encoding='utf-8') as f:
        classified_results = json.load(f)
    
    # Add the 'toxicity' attribute to each node based on the classified results
    for node in graph.nodes():
        graph.nodes[node]['toxicity'] = classified_results.get(str(node), "non-toxic")

def visualize_classified_network(graph, node_label_interval=10):
    color_map = []
    for node in graph.nodes():
        if graph.nodes[node]['toxicity'] == 'toxic':
            color_map.append('red')
        else:
            color_map.append('green')
    
    plt.figure(figsize=(30, 30)) 
    pos = nx.spring_layout(graph, k=2)  
    nx.draw_networkx_nodes(graph, pos, node_color=color_map, node_size=50)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=0.5)
    labels = {node: node for idx, node in enumerate(graph.nodes()) if idx % node_label_interval == 0}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.savefig('classified_information_diffusion_network.png')

def plot_degree_distribution(graph):
    degrees = [deg for _, deg in graph.degree()] #Gets degree of all nodes in graph
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20, color='blue', edgecolor='black')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig('degreeDistribution.png')

def plot_clustering_coefficient(graph):
    clustering_coeffs = nx.clustering(graph).values()  # Get clustering coefficients for all nodes
    plt.figure(figsize=(10, 6))
    plt.hist(list(clustering_coeffs), bins=20, color='green', edgecolor='black')
    plt.title('Clustering Coefficient Distribution')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.savefig('clusteringCoeff.png')

def calculate_average_friends(graph):
    degrees = dict(graph.degree())  

    local_average = dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True))
    top_node = next(iter(local_average.items()))
    
    global_average = sum(degrees.values()) / len(degrees)
    
    print(f"Global Average Number of Friends: {global_average}")
    print(f"Top node: {top_node[0]} with {top_node[1]} friends")
    print(f"Local Average: {local_average}")
    return local_average, global_average

all_posts = get_hashtag_posts()  # This gathers posts with the hashtag
seed_users = get_top_users() 
all_users = get_user_data(seed_users)  # This gathers data for specific users and their followers

info_diffusion_network = build_information_diffusion_network()
print(f"Number of nodes (posts): {info_diffusion_network.number_of_nodes()}")
print(f"Number of edges (interactions): {info_diffusion_network.number_of_edges()}")
plt.figure(figsize=(30, 30)) 
pos = nx.spring_layout(info_diffusion_network, k=2) 
labels = {node: node for idx, node in enumerate(info_diffusion_network.nodes()) if idx % 10 == 0}
nx.draw(info_diffusion_network, pos, with_labels=True, labels=labels, node_size=100, font_size=15)
nx.draw_networkx_edges(info_diffusion_network, pos, edge_color='gray', width=0.5)
plt.savefig('information_diffusion_network.png')

friendship_network = build_friendship_network()
print(f"Number of nodes (users): {friendship_network.number_of_nodes()}")
print(f"Number of edges (friendships): {friendship_network.number_of_edges()}")
plt.figure(figsize=(30, 30)) 
pos = nx.spring_layout(friendship_network, k=3) 
nx.draw(friendship_network, pos, with_labels=True, node_size=50, font_size=11, width=0.2) 
plt.savefig('friendship_network.png')

postContent = extract_post_content(info_diffusion_network)
ifToxic = classify_toxicity(postContent)
classify_nodes_in_network(info_diffusion_network)
visualize_classified_network(info_diffusion_network)
print("Done identifying Toxic Posts")

plot_degree_distribution(friendship_network)
plot_clustering_coefficient(friendship_network)
local_avg, global_avg = calculate_average_friends(friendship_network)