function [purity, nmi] = compute_purity_nmi_top(label, theta)

javaaddpath('.');

dlmwrite('./temp_label.txt',label);

[~,max_topic] = max(theta,[],2);

dlmwrite('./temp_topic.txt',max_topic);

results = ClusteringEval.evaluate('temp_label.txt','temp_topic.txt');

purity = results(1);
nmi = results(2);

end