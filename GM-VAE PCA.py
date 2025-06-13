# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
import torch

# %%
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Set random seeds (reproducibility)
np.random.seed(42)
tf.random.set_seed(42)

# %%
# Data preprocessing (train/validation/test splits)
def load_and_preprocess_data(file_path, train_size=0.6, val_size=0.2, test_size=0.2):
    df = pd.read_csv(file_path, sep=r'\s+')
    
    feature_cols = [f'C{i}' for i in range(1, 21)]
    X = df[feature_cols].values
    
    ids = df[['FID', 'IID.x']].copy()
    
    has_labels = 'SOL' in df.columns
    if has_labels:
        y = df['SOL'].values
    else:
        y = None
    
    X_temp, X_test, ids_temp, ids_test = train_test_split(
        X, ids, test_size=test_size, random_state=42, shuffle=True
    )
    
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, ids_train, ids_val = train_test_split(
        X_temp, ids_temp, test_size=val_size_adjusted, random_state=42, shuffle=True
    )
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    print(f"Data shapes: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}, Test={X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, X_scaled, scaler, ids, y, feature_cols

# %%
# Beta annealing callback (KL weight scheduling)
class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_var, beta_start=0.0, beta_end=1.0, annealing_epochs=100, warmup_epochs=20, verbose=0):
        super(BetaScheduler, self).__init__()
        self.beta_var = beta_var
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.annealing_epochs = annealing_epochs
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            new_beta = self.beta_start
        else:
            progress = min(1.0, (epoch - self.warmup_epochs) / self.annealing_epochs)
            new_beta = self.beta_start + progress * (self.beta_end - self.beta_start)
            new_beta = min(self.beta_end, new_beta)
        
        tf.keras.backend.set_value(self.beta_var, new_beta)
        
        if self.verbose:
            print(f"\nEpoch {epoch+1}: beta = {new_beta:.4f}")

# %%
# Custom checkpoint callback (saves on significant improvements)
class BestModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', min_delta=0.01, **kwargs):
        self.min_delta = min_delta
        self.best_value = np.inf
        super(BestModelCheckpoint, self).__init__(filepath, monitor=monitor, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is not None and (self.best_value - current) > self.min_delta:
            self.best_value = current
            super(BestModelCheckpoint, self).on_epoch_end(epoch, logs)

# %%
# VAE model creation (GMM prior, regularization)
def create_vae_gmm(input_dim=20, latent_dim=2, hidden_units=[64, 32], 
                  num_components=5, beta_start=0.0, l2_reg=0.0025, dropout_rate=0.35, learning_rate=0.001):
    
    beta = tf.Variable(beta_start, dtype=tf.float32, trainable=False, name='beta')
    
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dim,), name='encoder_input')
    x = encoder_inputs
    
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, 
                        activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=f'encoder_dense_{i}')(x)
        x = layers.BatchNormalization(name=f'encoder_bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i}')(x)
    
    z_mean = layers.Dense(latent_dim, name='z_mean',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var',
                            kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Reparameterization
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = latent_inputs
    
    for i, units in enumerate(reversed(hidden_units)):
        x = layers.Dense(units, 
                         activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name=f'decoder_dense_{i}')(x)
        x = layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i}')(x)
    
    decoder_outputs = layers.Dense(input_dim, name='decoder_output')(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    
    outputs = decoder(z)
    vae = Model(encoder_inputs, outputs, name='vae')
    
    # GMM initialization (circular for 2D)
    if latent_dim == 2:
        radius = 0.8  # adjust radius value
        theta = np.linspace(0, 2*np.pi, num_components, endpoint=False)
        gmm_means = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    else:
        gmm_means = []
        for i in range(num_components):
            mean = np.random.normal(0, 0.5, latent_dim)  # adjust initialization scale
            mean = mean / np.sqrt(np.sum(mean**2)) * 2.0  # adjust normalization factor
            gmm_means.append(mean)
        gmm_means = np.array(gmm_means)
    
    gmm_covs = [np.eye(latent_dim) * 0.5 for _ in range(num_components)]  # adjust covariance scale
    gmm_weights = np.ones(num_components) / num_components
    
    # GMM KL loss
    def gmm_kl_loss(y_true, y_pred):
        z_mean_batch, z_log_var_batch, z_batch = encoder(y_true)
        
        components = []
        for i in range(num_components):
            mean = tf.constant(gmm_means[i], dtype=tf.float32)
            cov = tf.constant(gmm_covs[i], dtype=tf.float32)
            
            mvn = tfd.MultivariateNormalFullCovariance(
                loc=mean,
                covariance_matrix=cov
            )
            components.append(mvn)
        
        gmm_prior = tfd.Mixture(
            cat=tfd.Categorical(probs=tf.constant(gmm_weights, dtype=tf.float32)),
            components=components
        )
        
        z_batch_tensor = tf.convert_to_tensor(z_batch)
        log_prob_gmm = gmm_prior.log_prob(z_batch_tensor)
        
        q_z = tfd.MultivariateNormalDiag(
            loc=z_mean_batch, 
            scale_diag=tf.exp(0.5 * z_log_var_batch)
        )
        log_prob_q = q_z.log_prob(z_batch_tensor)
        
        kl_gmm = tf.reduce_mean(log_prob_q - log_prob_gmm)
        
        kl_scale = 1.0 / input_dim
        kl_gmm = tf.clip_by_value(kl_gmm, -100.0, 100.0)
        return kl_scale * kl_gmm
    
    # Reconstruction loss
    def reconstruction_loss(y_true, y_pred):
        rec_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return tf.reduce_mean(rec_loss)
    
    # Total loss (reconstruction + beta * KL)
    def vae_gmm_loss(y_true, y_pred):
        rec_loss = reconstruction_loss(y_true, y_pred)
        kl = gmm_kl_loss(y_true, y_pred)
        return rec_loss + beta * kl
    
    def metric_reconstruction_loss(y_true, y_pred):
        return reconstruction_loss(y_true, y_pred)
    
    def metric_kl_loss(y_true, y_pred):
        return gmm_kl_loss(y_true, y_pred)
    
    def metric_beta(y_true, y_pred):
        return beta
    
    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=vae_gmm_loss,
        metrics=[
            metric_reconstruction_loss,
            metric_kl_loss,
            metric_beta
        ]
    )
    
    return vae, encoder, decoder, gmm_kl_loss, reconstruction_loss, vae_gmm_loss, beta

# %%
# Optimal cluster detection (multiple metrics)
def find_optimal_clusters(X, max_clusters=10, min_clusters=2, penalty_weight=0.2):
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    bic_scores = []
    aic_scores = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
    n_samples = X.shape[0]
    max_allowable = min(max_clusters, n_samples // 10)  # adjust minimum samples per cluster
    max_clusters = max(min_clusters, max_allowable)
    
    for n_components in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(
            n_components=n_components, 
            random_state=42, 
            covariance_type='full',
            n_init=10  # adjust initialization attempts
        )
        gmm.fit(X)
        
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        
        penalized_bic = bic + penalty_weight * n_components * np.log(X.shape[0])
        bic_scores.append(penalized_bic)
        aic_scores.append(aic)
        
        labels = gmm.predict(X)
        
        if n_components >= 2:
            sil = silhouette_score(X, labels)
            silhouette_scores.append(sil)
            
            ch = calinski_harabasz_score(X, labels)
            ch_scores.append(ch)
            
            db = davies_bouldin_score(X, labels)
            db_scores.append(db)
        else:
            silhouette_scores.append(0)
            ch_scores.append(0)
            db_scores.append(float('inf'))
        
        print(f"Clusters: {n_components}, BIC: {penalized_bic:.2f}, " +
              f"Silhouette: {silhouette_scores[-1]:.4f}, " +
              f"Calinski-Harabasz: {ch_scores[-1]:.2f}, " +
              f"Davies-Bouldin: {db_scores[-1]:.4f}")
    
    # Normalize scores
    def safe_normalize(values, invert=False):
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            normalized = [0.5] * len(values)
        else:
            normalized = [(x - min_val) / (max_val - min_val) for x in values]
        
        if invert:
            normalized = [1 - x for x in normalized]
        
        return normalized
    
    norm_bic = safe_normalize(bic_scores, invert=True)
    norm_sil = safe_normalize(silhouette_scores)
    norm_ch = safe_normalize(ch_scores)
    norm_db = safe_normalize(db_scores, invert=True)
    
    # Weighted combination (adjust weights as needed)
    weights = {
        'bic': 0.25,
        'silhouette': 0.25,
        'calinski_harabasz': 0.25,
        'davies_bouldin': 0.25
    }
    
    combined_scores = []
    for i in range(len(norm_bic)):
        idx = max(0, i - (min_clusters - 1))
        
        if i == 0 and min_clusters == 1:
            score = weights['bic'] * norm_bic[i]
        else:
            score = (
                weights['bic'] * norm_bic[i] +
                weights['silhouette'] * norm_sil[idx] +
                weights['calinski_harabasz'] * norm_ch[idx] +
                weights['davies_bouldin'] * norm_db[idx]
            )
        combined_scores.append(score)
    
    optimal_clusters = combined_scores.index(max(combined_scores)) + min_clusters
    
    # Visualization
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(min_clusters, max_clusters + 1), bic_scores, marker='o', color='blue')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', 
                label=f'Optimal Clusters: {optimal_clusters}')
    plt.title('BIC Score (lower is better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Penalized BIC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', color='green')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', 
                label=f'Optimal Clusters: {optimal_clusters}')
    plt.title('Silhouette Score (higher is better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(range(min_clusters, max_clusters + 1), ch_scores, marker='o', color='purple')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', 
                label=f'Optimal Clusters: {optimal_clusters}')
    plt.title('Calinski-Harabasz Index (higher is better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(range(min_clusters, max_clusters + 1), db_scores, marker='o', color='orange')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', 
                label=f'Optimal Clusters: {optimal_clusters}')
    plt.title('Davies-Bouldin Index (lower is better)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f'Cluster Evaluation Metrics (Optimal clusters: {optimal_clusters})', 
                 fontsize=16, y=1.02)
    
    metrics = {
        'n_components': list(range(min_clusters, max_clusters + 1)),
        'bic': bic_scores,
        'silhouette': silhouette_scores,
        'calinski_harabasz': ch_scores,
        'davies_bouldin': db_scores,
        'combined_score': combined_scores,
        'optimal_clusters': optimal_clusters
    }
    
    return optimal_clusters, fig, metrics

# %%
# GMM fitting to latent space
def fit_gmm_to_latent(encoder, X, n_components, covariance_type='full'):
    z_mean, z_log_var, _ = encoder.predict(X, verbose=0)
    
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(
        n_components=n_components, 
        random_state=42, 
        covariance_type=covariance_type,
        n_init=20  # adjust initialization attempts
    )
    gmm.fit(z_mean)
    
    labels = gmm.predict(z_mean)
    probs = gmm.predict_proba(z_mean)
    
    point_entropies = -np.sum(probs * np.log2(probs + 1e-10), axis=1)
    
    return gmm, labels, z_mean, probs, point_entropies

# %%
# Latent space visualization
def visualize_latent_space_multiple(z_mean, labels, n_components, entropies=None, gmm=None, probs=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    cmap = plt.cm.get_cmap('tab10', n_components)
    
    scatter1 = axes[0].scatter(z_mean[:, 0], z_mean[:, 1], c=labels, 
                             cmap=cmap, alpha=0.8, s=25, edgecolors='k', linewidths=0.3)
    axes[0].set_title(f'Latent Space with {n_components} GMM Clusters')
    axes[0].set_xlabel('Latent Dimension 1')
    axes[0].set_ylabel('Latent Dimension 2')
    axes[0].grid(alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    if entropies is not None:
        scatter2 = axes[1].scatter(z_mean[:, 0], z_mean[:, 1], c=entropies, 
                                 cmap='viridis', alpha=0.8, s=25, edgecolors='k', linewidths=0.3)
        axes[1].set_title('Clustering Uncertainty (Entropy)')
        axes[1].set_xlabel('Latent Dimension 1')
        axes[1].set_ylabel('Latent Dimension 2')
        axes[1].grid(alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Entropy')
    elif n_components > 1 and gmm is not None:
        x_min, x_max = z_mean[:, 0].min() - 1, z_mean[:, 0].max() + 1
        y_min, y_max = z_mean[:, 1].min() - 1, z_mean[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        try:
            Z = -gmm.score_samples(mesh_points)
            Z = Z.reshape(xx.shape)
            
            contour = axes[1].contourf(xx, yy, Z, cmap='viridis_r', alpha=0.5, levels=np.linspace(Z.min(), Z.max(), 15))
            plt.colorbar(contour, ax=axes[1], label='Negative Log-Likelihood')
            
            axes[1].scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap=cmap, 
                         alpha=0.8, s=25, edgecolors='k', linewidths=0.5)
            axes[1].set_title('Density Contours with Data Points')
            axes[1].set_xlabel('Latent Dimension 1')
            axes[1].set_ylabel('Latent Dimension 2')
            axes[1].grid(alpha=0.3)
        except:
            axes[1].scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap=cmap, 
                         alpha=0.8, s=25)
            axes[1].set_title('Latent Space (Alternative View)')
            axes[1].set_xlabel('Latent Dimension 1')
            axes[1].set_ylabel('Latent Dimension 2')
            axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

# %%
# Clustering evaluation
def evaluate_clustering(labels, z_mean, X_original=None):
    from sklearn import metrics
    
    if len(np.unique(labels)) > 1:
        silhouette = metrics.silhouette_score(z_mean, labels)
        db_index = metrics.davies_bouldin_score(z_mean, labels)
        ch_index = metrics.calinski_harabasz_score(z_mean, labels)
        
        if X_original is not None:
            silhouette_orig = metrics.silhouette_score(X_original, labels)
            db_index_orig = metrics.davies_bouldin_score(X_original, labels)
            ch_index_orig = metrics.calinski_harabasz_score(X_original, labels)
            
            print("\nClustering metrics in latent space:")
            print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range -1 to 1)")
            print(f"  Davies-Bouldin Index: {db_index:.4f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {ch_index:.4f} (higher is better)")
            
            print("\nClustering metrics in original space:")
            print(f"  Silhouette Score: {silhouette_orig:.4f} (higher is better, range -1 to 1)")
            print(f"  Davies-Bouldin Index: {db_index_orig:.4f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {ch_index_orig:.4f} (higher is better)")
            
            return {
                'latent': {
                    'silhouette': silhouette,
                    'davies_bouldin': db_index,
                    'calinski_harabasz': ch_index
                },
                'original': {
                    'silhouette': silhouette_orig,
                    'davies_bouldin': db_index_orig,
                    'calinski_harabasz': ch_index_orig
                }
            }
        else:
            print(f"Silhouette Score: {silhouette:.4f} (higher is better, range -1 to 1)")
            print(f"Davies-Bouldin Index: {db_index:.4f} (lower is better)")
            print(f"Calinski-Harabasz Index: {ch_index:.4f} (higher is better)")
            
            return {
                'silhouette': silhouette,
                'davies_bouldin': db_index,
                'calinski_harabasz': ch_index
            }
    else:
        print("Only one cluster found. Cannot calculate clustering metrics.")
        return None

# %%
# Bootstrap validation (cluster stability)
def bootstrap_cluster_validation(encoder, X_data, 
                               n_bootstraps=100, 
                               optimal_clusters=2,
                               covariance_type='full',
                               random_seed=42):
    from sklearn.mixture import GaussianMixture
    import numpy as np
    from sklearn.utils import resample
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    
    np.random.seed(random_seed)
    n_samples = X_data.shape[0]
    
    z_mean_full, _, _ = encoder.predict(X_data, verbose=0)
    gmm_full = GaussianMixture(
        n_components=optimal_clusters, 
        random_state=random_seed, 
        covariance_type=covariance_type,
        n_init=20  # adjust initialization attempts
    )
    gmm_full.fit(z_mean_full)
    labels_full = gmm_full.predict(z_mean_full)
    
    rand_scores = []
    ami_scores = []
    cluster_counts = []
    sample_stability = np.zeros(n_samples)
    cluster_membership = np.zeros((n_samples, optimal_clusters))
    
    for i in range(n_bootstraps):
        if i % 10 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstraps}")
            
        indices = resample(range(n_samples), replace=True, n_samples=n_samples, random_state=i)
        X_bootstrap = X_data[indices]
        
        z_mean_boot, _, _ = encoder.predict(X_bootstrap, verbose=0)
        
        from sklearn.metrics import silhouette_score
        bic_scores = []
        sil_scores = []
        
        for k in range(max(2, optimal_clusters-1), optimal_clusters+2):
            gmm_k = GaussianMixture(
                n_components=k, 
                random_state=random_seed, 
                covariance_type=covariance_type,
                n_init=10  # adjust initialization attempts
            )
            gmm_k.fit(z_mean_boot)
            bic_scores.append(gmm_k.bic(z_mean_boot))
            
            if k >= 2:
                labels_k = gmm_k.predict(z_mean_boot)
                try:
                    sil_scores.append(silhouette_score(z_mean_boot, labels_k))
                except:
                    sil_scores.append(0)
        
        boot_opt_clusters = bic_scores.index(min(bic_scores)) + max(2, optimal_clusters-1)
        cluster_counts.append(boot_opt_clusters)
        
        gmm_boot = GaussianMixture(
            n_components=optimal_clusters, 
            random_state=random_seed, 
            covariance_type=covariance_type,
            n_init=20  # adjust initialization attempts
        )
        gmm_boot.fit(z_mean_boot)
        
        z_mean_all, _, _ = encoder.predict(X_data, verbose=0)
        labels_boot = gmm_boot.predict(z_mean_all)
        
        rand_scores.append(adjusted_rand_score(labels_full, labels_boot))
        ami_scores.append(adjusted_mutual_info_score(labels_full, labels_boot))
        
        for j in range(n_samples):
            if labels_boot[j] == labels_full[j]:
                sample_stability[j] += 1
            cluster_membership[j, labels_boot[j]] += 1
    
    sample_stability /= n_bootstraps
    cluster_membership /= n_bootstraps
    
    cluster_entropy = -np.sum(
        cluster_membership * np.log2(cluster_membership + 1e-10), 
        axis=1
    )
    
    cluster_consistency = {}
    for i in range(optimal_clusters):
        points_in_cluster = (labels_full == i)
        if np.sum(points_in_cluster) > 0:
            cluster_consistency[i] = np.mean(sample_stability[points_in_cluster])
        else:
            cluster_consistency[i] = 0
    
    cluster_count_freq = np.bincount(cluster_counts)
    most_common_count = np.argmax(cluster_count_freq)
    
    stability_results = {
        'mean_adjusted_rand_index': np.mean(rand_scores),
        'mean_adjusted_mutual_info': np.mean(ami_scores),
        'cluster_consistency': cluster_consistency,
        'sample_stability': sample_stability,
        'cluster_entropy': cluster_entropy,
        'most_common_cluster_count': most_common_count,
        'cluster_count_frequency': {i: count for i, count in enumerate(cluster_count_freq) if count > 0},
        'cluster_membership_probability': cluster_membership
    }
    
    return stability_results, labels_full

# %%
# Model architecture search (hyperparameter tuning)
def model_architecture_search(file_path, latent_dim=2, 
                            depths=[1, 2, 3], 
                            widths=[32, 64, 128], 
                            betas=[0.1, 0.5, 1.0],
                            dropout_rates=[0.1, 0.2, 0.3],
                            batch_size=32, epochs=30,
                            save_dir='./vae_search_results'):
    import os
    import json
    from itertools import product
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, X_scaled, scaler, ids, y, feature_cols = load_and_preprocess_data(file_path)
    
    search_results = []
    best_val_loss = float('inf')
    best_params = None
    
    init_components, _ = find_optimal_clusters(
        X_scaled, 
        max_clusters=10,  # adjust max clusters for search
        min_clusters=2,
        penalty_weight=0.3  # adjust penalty weight
    )
    plt.close()
    
    total_combinations = len(depths) * len(widths) * len(betas) * len(dropout_rates)
    current_combo = 0
    
    for depth, width, beta_end, dropout_rate in product(depths, widths, betas, dropout_rates):
        current_combo += 1
        print(f"\nTrying combination {current_combo}/{total_combinations}:")
        print(f"Depth={depth}, Width={width}, Beta={beta_end}, Dropout={dropout_rate}")
        
        hidden_units = [width] * depth
        
        vae, encoder, decoder, _, _, _, beta_var = create_vae_gmm(
            input_dim=X_train.shape[1], 
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            num_components=init_components,
            beta_start=0.0,  # adjust start value
            l2_reg=0.001,  # adjust regularization
            dropout_rate=dropout_rate,
            learning_rate=0.001  # adjust learning rate
        )
        
        beta_scheduler = BetaScheduler(
            beta_var=beta_var,
            beta_start=0.0,  # adjust start value
            beta_end=beta_end,  
            annealing_epochs=int(epochs * 0.7),  # adjust annealing proportion
            warmup_epochs=3,  # adjust warmup epochs
            verbose=0   
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # adjust patience
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,  # adjust reduction factor
            patience=5,  # adjust patience
            min_lr=0.0001,  # adjust minimum learning rate
            verbose=0
        )

        search_callbacks = [beta_scheduler, early_stopping, reduce_lr]
        
        history = vae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=search_callbacks,
            verbose=0
        )
        
        val_loss = min(history.history['val_loss'])
        final_epoch = len(history.history['val_loss'])
        
        result = {
            'depth': depth,
            'width': width,
            'hidden_units': hidden_units,
            'beta_end': beta_end,
            'dropout_rate': dropout_rate,
            'val_loss': val_loss,
            'epochs_trained': final_epoch,
            'early_stopped': final_epoch < epochs
        }
        
        search_results.append(result)
        
        print(f"Val Loss: {val_loss:.4f}, Epochs: {final_epoch}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = result
            print("New best model found!")
        
        tf.keras.backend.clear_session()
    
    search_results = sorted(search_results, key=lambda x: x['val_loss'])
    
    with open(f"{save_dir}/search_results.json", 'w') as f:
        json.dump(search_results, f, indent=2)
    
    print("\nSearch completed. Best parameters:")
    print(f"Depth: {best_params['depth']}")
    print(f"Width: {best_params['width']}")
    print(f"Beta: {best_params['beta_end']}")
    print(f"Dropout: {best_params['dropout_rate']}")
    print(f"Validation Loss: {best_params['val_loss']:.4f}")
    
    return search_results, best_params

# %%
# Overfitting evaluation
def evaluate_overfitting(vae, encoder, X_train, X_val):
    train_recon = vae.predict(X_train, verbose=0)
    val_recon = vae.predict(X_val, verbose=0)
    
    train_mse = np.mean(np.square(X_train - train_recon))
    val_mse = np.mean(np.square(X_val - val_recon))
    
    print(f"\nTraining MSE: {train_mse:.6f}")
    print(f"Validation MSE: {val_mse:.6f}")
    print(f"Difference (Val - Train): {val_mse - train_mse:.6f}")
    
    mse_ratio = val_mse / train_mse
    print(f"Validation/Training MSE Ratio: {mse_ratio:.4f}")
    
    if mse_ratio > 1.5:  # adjust threshold
        print("WARNING: Possible overfitting detected (val/train ratio > 1.5)")
        print("Recommendations:")
        print("1. Reduce model complexity (fewer hidden units)")
        print("2. Increase regularization (higher beta, L2 weight)")
        print("3. Consider reducing max_clusters")
        print("4. Use more dropout or reduce epochs")
    elif mse_ratio > 1.2:  # adjust threshold
        print("CAUTION: Some overfitting may be occurring (val/train ratio > 1.2)")
    else:
        print("Model does not appear to be overfitting significantly")
        
    z_mean_train, _, _ = encoder.predict(X_train, verbose=0)
    
    latent_var = np.var(z_mean_train, axis=0)
    print("\nVariance explained by each latent dimension:")
    for i, var in enumerate(latent_var):
        print(f"Dimension {i+1}: {var:.4f} ({var/np.sum(latent_var)*100:.2f}%)")
    
    return {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'mse_ratio': mse_ratio,
        'latent_var': latent_var
    }

# %%
# Loss visualization
def plot_loss_components(history):
    plt.figure(figsize=(12, 8))
    
    rec_loss_key = next((key for key in history.keys() if 'reconstruction' in key), None)
    kl_loss_key = next((key for key in history.keys() if 'kl' in key), None)
    beta_key = next((key for key in history.keys() if 'beta' in key), None)
    
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    
    if len(history['loss']) > 10:
        from scipy.signal import savgol_filter
        y_smooth = savgol_filter(history['loss'], min(21, len(history['loss']) // 3 * 2 + 1), 3)
        plt.plot(y_smooth, 'k--', linewidth=1, alpha=0.5, label='Smoothed Trend')
    
    plt.title('Total Loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    
    if rec_loss_key:
        plt.plot(history[rec_loss_key], label='Reconstruction Loss', color='green', linewidth=2)
    
    if kl_loss_key:
        plt.plot(history[kl_loss_key], label='KL Divergence (GMM)', color='purple', linewidth=2)
    
    if beta_key:
        plt.plot(history[beta_key], label='Beta (KL Weight)', color='orange', linewidth=2)
        
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

# %%
# Learning rate scheduler (warmup + cosine decay)
class WarmupCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs=5, hold_base_epochs=0,
                 verbose=0):
        super(WarmupCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.hold_base_epochs = hold_base_epochs
        self.verbose = verbose
        self.learning_rates = []
    
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.calculate_lr(epoch)
        
        try:
            self.model.optimizer._learning_rate.assign(lr)
        except AttributeError:
            try:
                tf.keras.backend.set_value(self.model.optimizer._learning_rate, lr)
            except AttributeError:
                try:
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                except AttributeError:
                    print("Warning: Could not set learning rate - scheduler may not work")
        
        if self.verbose:
            print(f"\nEpoch {epoch+1}: learning rate set to {lr:.6f}")
    
    def calculate_lr(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        elif epoch < self.warmup_epochs + self.hold_base_epochs:
            lr = self.learning_rate_base
        else:
            progress = (epoch - self.warmup_epochs - self.hold_base_epochs) / \
                      max(1, self.total_epochs - self.warmup_epochs - self.hold_base_epochs)
            progress = min(1.0, progress)
            lr = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * progress))
        
        return lr
    
    def on_epoch_end(self, epoch, logs=None):
        try:
            current_lr = self.calculate_lr(epoch)
            self.learning_rates.append(current_lr)
        except:
            pass

# %%
# Main pipeline (complete VAE-GMM workflow)
def run_vae_gmm_pipeline(file_path, latent_dim=2, hidden_units=[64, 32], 
                        min_clusters=1, max_clusters=8, batch_size=32, epochs=100,
                        beta_start=0.0, beta_end=1.0, l2_reg=0.001, dropout_rate=0.2, learning_rate=0.001,
                        show_plots=True, save_dir='./vae_results'):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, X_scaled, scaler, ids, y, feature_cols = load_and_preprocess_data(file_path)
    
    print("Estimating initial number of GMM components for prior...")
    init_components, bic_plot, init_metrics = find_optimal_clusters(
        X_scaled, 
        max_clusters=min(10, max_clusters),  # adjust value
        min_clusters=min_clusters,
        penalty_weight=0.3  # adjust penalty weight
    )
    print(f"Initial estimate of GMM components: {init_components}")
    
    if show_plots:
        plt.savefig(f"{save_dir}/initial_cluster_estimate.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.close()
    
    print(f"Creating VAE model with GMM prior (latent_dim={latent_dim}, GMM components={init_components})...")
    vae, encoder, decoder, kl_loss_fn, recon_loss_fn, total_loss_fn, beta_var = create_vae_gmm(
        input_dim=X_train.shape[1], 
        latent_dim=latent_dim,
        hidden_units=hidden_units,
        num_components=init_components,
        beta_start=beta_start,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate  # adjust value
    )
    
    vae.summary()
    
    # Callbacks
    beta_scheduler = BetaScheduler(
        beta_var=beta_var,
        beta_start=beta_start,
        beta_end=beta_end,  
        annealing_epochs=epochs * 3//4,  # adjust proportion
        warmup_epochs=10,  # adjust warmup epochs
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,  # adjust patience
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = BestModelCheckpoint(
        f"{save_dir}/best_vae_model.h5",
        monitor='val_loss',
        min_delta=0.01,  # adjust improvement threshold
        save_best_only=True,
        verbose=1
    )
    
    lr_scheduler = WarmupCosineDecayScheduler(
        learning_rate_base=learning_rate,  # adjust base learning rate
        total_epochs=epochs,
        warmup_epochs=5,  # adjust warmup epochs
        hold_base_epochs=10,  # adjust hold epochs
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,  # adjust reduction factor
        patience=8,  # adjust patience
        min_lr=0.0001,  # adjust minimum learning rate
        verbose=1
    )

    callbacks = [beta_scheduler, early_stopping, checkpoint, lr_scheduler, reduce_lr]
    
    print("Training VAE...")
    history = vae.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEvaluating model for overfitting...")
    overfitting_metrics = evaluate_overfitting(vae, encoder, X_train, X_val)
    
    with open(f"{save_dir}/overfitting_metrics.txt", 'w') as f:
        for key, value in overfitting_metrics.items():
            if isinstance(value, np.ndarray):
                f.write(f"{key}: {list(value)}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print("Plotting loss history...")
    loss_plot = plot_loss_components(history.history)
    if show_plots:
        plt.savefig(f"{save_dir}/training_loss.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.close()
    
    print("Extracting latent space representation...")
    z_mean, z_log_var, _ = encoder.predict(X_scaled, verbose=0)
    
    print("Finding optimal number of clusters in latent space...")
    optimal_clusters, bic_plot, cluster_metrics_data = find_optimal_clusters(
        z_mean, 
        max_clusters=max_clusters, 
        min_clusters=min_clusters
    )
    print(f"Optimal number of clusters in latent space: {optimal_clusters}")
    
    if show_plots:
        plt.savefig(f"{save_dir}/optimal_clusters.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.close()
    
    print(f"Fitting GMM with {optimal_clusters} clusters...")
    gmm, labels, z_latent, probs, entropies = fit_gmm_to_latent(encoder, X_scaled, optimal_clusters)
    
    print("Evaluating clustering quality...")
    cluster_metrics = evaluate_clustering(labels, z_latent, X_scaled)
    
    with open(f"{save_dir}/cluster_metrics.txt", 'w') as f:
        f.write(f"Number of clusters: {optimal_clusters}\n\n")
        
        if isinstance(cluster_metrics, dict) and 'latent' in cluster_metrics:
            f.write("Metrics in latent space:\n")
            for key, value in cluster_metrics['latent'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nMetrics in original space:\n")
            for key, value in cluster_metrics['original'].items():
                f.write(f"  {key}: {value}\n")
        elif isinstance(cluster_metrics, dict):
            for key, value in cluster_metrics.items():
                f.write(f"{key}: {value}\n")
    
    metrics_df = pd.DataFrame({
        'n_components': cluster_metrics_data['n_components'],
        'bic': cluster_metrics_data['bic'],
        'silhouette': cluster_metrics_data['silhouette'],
        'calinski_harabasz': cluster_metrics_data['calinski_harabasz'],
        'davies_bouldin': cluster_metrics_data['davies_bouldin'],
        'combined_score': cluster_metrics_data['combined_score']
    })
    metrics_df.to_csv(f"{save_dir}/cluster_metrics_comparison.csv", index=False)
    
    if optimal_clusters >= 2:
        print("\nPerforming bootstrap validation to assess cluster stability...")
        print("This may take a few minutes...")
        stability_results, original_labels = bootstrap_cluster_validation(
            encoder=encoder,
            X_data=X_scaled,
            n_bootstraps=100,  # adjust number of bootstrap iterations
            optimal_clusters=optimal_clusters,
            covariance_type='full',
            random_seed=42
        )
        
        print(f"\nCluster stability assessment results:")
        print(f"Mean Adjusted Rand Index: {stability_results['mean_adjusted_rand_index']:.4f}")
        print(f"Mean Adjusted Mutual Information: {stability_results['mean_adjusted_mutual_info']:.4f}")
        print(f"Most frequent cluster count in bootstraps: {stability_results['most_common_cluster_count']}")
        
        print(f"\nCluster consistency scores:")
        for cluster, score in stability_results['cluster_consistency'].items():
            print(f"  Cluster {cluster}: {score:.4f} (stability)")
        
        with open(f"{save_dir}/stability_metrics.txt", 'w') as f:
            f.write(f"Mean Adjusted Rand Index: {stability_results['mean_adjusted_rand_index']:.4f}\n")
            f.write(f"Mean Adjusted Mutual Information: {stability_results['mean_adjusted_mutual_info']:.4f}\n")
            f.write(f"Most frequent cluster count: {stability_results['most_common_cluster_count']}\n\n")
            
            f.write("Cluster consistency scores:\n")
            for cluster, score in stability_results['cluster_consistency'].items():
                f.write(f"  Cluster {cluster}: {score:.4f}\n")
            
            f.write("\nCluster count frequencies:\n")
            for count, freq in stability_results['cluster_count_frequency'].items():
                f.write(f"  {count} clusters: {freq} times ({freq/100:.1f}%)\n")
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(stability_results['sample_stability'], bins=20, alpha=0.7)
        plt.title('Distribution of Sample Stability')
        plt.xlabel('Stability Score (higher = more stable)')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(
            z_latent[:, 0], z_latent[:, 1], 
            c=stability_results['sample_stability'], 
            cmap='viridis', 
            alpha=0.8, 
            s=30, 
            edgecolors='k', 
            linewidths=0.3
        )
        plt.colorbar(scatter, label='Stability Score')
        plt.title('Cluster Stability in Latent Space')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(stability_results['cluster_entropy'], bins=20, alpha=0.7, color='orange')
        plt.title('Distribution of Cluster Assignment Entropy')
        plt.xlabel('Entropy (lower = more consistent)')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        stability_plot = plt.gcf()
        plt.savefig(f"{save_dir}/cluster_stability.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        print("\nSkipping bootstrap validation since fewer than 2 clusters were found.")
        stability_results = None
        stability_plot = None
    
    print("Creating latent space visualization...")
    latent_plot = visualize_latent_space_multiple(z_latent, labels, optimal_clusters, entropies, gmm, probs)
    if show_plots:
        plt.savefig(f"{save_dir}/latent_space.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.close()
    
    result_df = ids.copy()
    result_df['Cluster'] = labels
    
    result_df['LatentDim1'] = z_latent[:, 0]
    result_df['LatentDim2'] = z_latent[:, 1]
    
    for i in range(optimal_clusters):
        result_df[f'Prob_Cluster_{i}'] = probs[:, i]
    
    result_df['Uncertainty'] = entropies
    
    if optimal_clusters >= 2 and stability_results is not None:
        result_df['Sample_Stability'] = stability_results['sample_stability']
        result_df['Cluster_Entropy'] = stability_results['cluster_entropy']
    
    result_df.to_csv(f"{save_dir}/cluster_assignments.csv", index=False)
    
    return {
        'vae': vae,
        'encoder': encoder,
        'decoder': decoder,
        'gmm': gmm,
        'labels': labels,
        'optimal_clusters': optimal_clusters,
        'result_df': result_df,
        'cluster_metrics': cluster_metrics,
        'stability_results': stability_results if optimal_clusters >= 2 else None,
        'overfitting_metrics': overfitting_metrics,
        'history': history.history,
        'z_latent': z_latent,
        'probs': probs,
        'entropies': entropies,
        'plots': {
            'initial_clusters_plot': bic_plot,
            'latent_plot': latent_plot,
            'loss_plot': loss_plot,
            'stability_plot': stability_plot if optimal_clusters >= 2 else None
        }
    }

# %%
# Main execution
if __name__ == "__main__":
    # Configuration
    file_path = "scaled_cases.txt"  # replace with your data file
    save_dir = "VAE_PCA1_cases_results"  # adjust output directory
    
    # Run pipeline (adjust hyperparameters as needed)
    results = run_vae_gmm_pipeline(
        file_path=file_path,
        latent_dim=2,                # adjust latent dimensions
        hidden_units=[64, 32],       # adjust network architecture [layer1_units, layer2_units, ...]
        min_clusters=1,              # adjust minimum clusters
        max_clusters=21,             # adjust maximum clusters
        batch_size=32,               # adjust batch size
        epochs=100,                  # adjust training epochs
        beta_start=0.0,              # adjust KL weight start
        beta_end=0.5,                # adjust KL weight end
        l2_reg=0.001,                # adjust L2 regularization
        dropout_rate=0.2,            # adjust dropout rate
        show_plots=True,             # set to False to suppress plots
        save_dir=save_dir,
        learning_rate=0.001          # adjust learning rate
    )
    
    # Results summary
    result_df = results['result_df']
    optimal_clusters = results['optimal_clusters']
    labels = results['labels']
    z_latent = results['z_latent']
    
    print(f"\nFound {optimal_clusters} clusters in the data")
    
    cluster_counts = result_df['Cluster'].value_counts().sort_index()
    print("\nCluster distribution:")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} samples ({count/len(result_df)*100:.1f}%)")
    
    print(f"\nDetailed cluster assignments saved to: {save_dir}/cluster_assignments.csv")
    
    if 'stability_results' in results and results['stability_results'] is not None:
        stability = results['stability_results']
        print("\nStability assessment results:")
        print(f"  Adjusted Rand Index: {stability['mean_adjusted_rand_index']:.4f}")
        print(f"  Most frequent cluster count in bootstraps: {stability['most_common_cluster_count']}")
        
        print("\nCluster stability scores:")
        for cluster, score in stability['cluster_consistency'].items():
            print(f"  Cluster {cluster}: {score:.4f} (stability)")
    
    # Final visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        z_latent[:, 0], 
        z_latent[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=0.8, 
        s=50, 
        edgecolors='k', 
        linewidths=0.5
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Latent Space Clustering ({optimal_clusters} clusters)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_clusters.png", dpi=300)
    plt.show()

