function [purities, nmis] = compute_purity_nmi_kmeans(label, theta, Ks)

javaaddpath('.');

purities = [];

nmis = [];

dlmwrite('./temp_label.txt',label);

if  sum(sum(theta, 2)) - length(theta) > 0.0001
    disp('softmax');
    theta = exp(theta) ./ sum(exp(theta), 2);    
end

for K = Ks
    
    
    idx = kmeans(theta, K, 'Distance', 'correlation');

    dlmwrite('./temp_topic.txt',idx);

    results = ClusteringEval.evaluate('temp_label.txt','temp_topic.txt');
    
    purities = [purities, results(1)];
    nmis = [nmis, results(2)];

end

end