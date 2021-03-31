function td = compute_topic_diversity(phi, top_word)
        
K = size(phi, 1);
list_w = [];
for k = 1:K
    [~, w_idx] = sort(phi(k, :), 'descend');
    list_w = [list_w; w_idx(1:top_word)];     
end

td = length(unique(list_w(:))) / (top_word * K);

end