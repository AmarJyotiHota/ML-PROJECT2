def create_eda_plots(df):
    """
    Create comprehensive EDA plots for the dataset
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    
    plots = []
    
    try:
        # 1. Distribution of Target Variable (if exists)
        if 'target' in df.columns or df.select_dtypes(include=[np.number]).shape[1] > 0:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = 'target' if 'target' in df.columns else numeric_cols[0]
                sns.histplot(df[target_col].dropna(), kde=True, ax=ax1)
                ax1.set_title(f'Distribution of {target_col}')
                ax1.set_xlabel(target_col)
                ax1.set_ylabel('Frequency')
                plt.tight_layout()
                plots.append(fig1)
        
        # 2. Correlation Matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            corr_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, ax=ax2)
            ax2.set_title('Feature Correlation Matrix')
            plt.tight_layout()
            plots.append(fig2)
        
        # 3. Missing Values Heatmap
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        missing_df = df.isnull()
        sns.heatmap(missing_df, cbar=False, yticklabels=False, 
                   cmap='viridis', ax=ax3)
        ax3.set_title('Missing Values Heatmap')
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Samples')
        plt.tight_layout()
        plots.append(fig3)
        
        # 4. Box Plots for Outliers Detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(6, len(numeric_cols))
            fig4, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:n_cols]):
                if i < len(axes):
                    df[col].dropna().plot(kind='box', ax=axes[i])
                    axes[i].set_title(f'Box Plot - {col}')
                    axes[i].set_ylabel(col)
            
            # Hide empty subplots
            for i in range(len(numeric_cols[:n_cols]), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Outliers Detection - Box Plots', y=1.02)
            plt.tight_layout()
            plots.append(fig4)
        
        # 5. Pairplot of Important Features
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 6:
            fig5 = plt.figure(figsize=(12, 10))
            important_cols = numeric_cols
