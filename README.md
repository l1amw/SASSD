# Generalizable Speech Spoofing Detection Against Silence Trimming with Data Augmentation and Multi-task Meta-Learning
A major difficulty in speech spoofing detection lies in improving the generalization ability to detect unknown forgery methods. However, most previous methods do not consider the interference of silence information on the generalization performance of speech spoofing detection. Notably, we experimentally observe that the generalization performance of existing methods drops sharply when silence segments are trimmed. This indicates that previous works have two problems: a) they do not remove the interference of silence and over-rely on silence information, and b) they lack the ability to uncover general forgery traces in utterance segments. To solve the above two problems, we propose a novel Silence-Agnostic Speech Spoofing Detection (SASSD) framework. To be specific, unlike previous methods trained on speech samples with silence information, we completely remove the leading and trailing silence segments from all speech samples to eliminate the interference of silence and focus on utterance information. Meanwhile, to uncover general forgery traces in utterance segments and improve the generalization ability, we view speech spoofing detection as a domain generalization problem and employ meta-learning to simulate the actual domain shift scenarios, which can reduce overfitting to specific forgery methods. In addition, to improve the domain generalization of meta-learning, a novel data augmentation method named ShuffleMix is proposed. Unlike previous methods that only consider inter-speech patterns, our method additionally introduces an intra-speech augmentation technique, which performs enhancements within a single speech and across multiple speech to generate more diverse forged samples. Extensive experiments show that our method achieves SOTA on the ASVspoof 2019LA dataset. In particular, our method achieves 0.231% EER and 2.529% EER on the original dataset with silence information and the silence-trimmed dataset, respectively.
