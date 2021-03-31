

save_dir = 'REPLACE WITH YOUR OWN SAVE DIR';
load([save_dir, '/save.mat']);

load './datasets/REPLACE WITH YOUR OWN DATASET/filtered_data.mat'

[top_purity, top_nmi] = compute_purity_nmi_top(labelsTest, test_theta)

[kmeans_purities, kmeans_nims] = compute_purity_nmi_kmeans(labelsTest, test_theta, [20, 40, 60, 80, 100])

topic_diversity_all_topics = compute_topic_diversity(phi, 25)





