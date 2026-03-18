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
            important_cols = numeric_cols[:min(5, len(numeric_cols))]
            if len(important_cols) >= 2:
                g = sns.pairplot(df[important_cols].dropna())
                g.fig.suptitle('Pairplot of Key Features', y=1.02)
                plots.append(g.fig)
        
        # 6. Categorical Features Distribution
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            n_cat_cols = min(4, len(cat_cols))
            fig6, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(cat_cols[:n_cat_cols]):
                if i < len(axes):
                    value_counts = df[col].value_counts().head(10)
                    axes[i].bar(range(len(value_counts)), value_counts.values)
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                    axes[i].set_title(f'Top Categories - {col}')
                    axes[i].set_ylabel('Count')
            
            # Hide empty subplots
            for i in range(len(cat_cols[:n_cat_cols]), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Categorical Features Distribution', y=1.02)
            plt.tight_layout()
            plots.append(fig6)
        
        # 7. Time Series Plot (if date column exists)
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            fig7, ax7 = plt.subplots(figsize=(12, 6))
            date_col = date_cols[0]
            numeric_col = numeric_cols[0]
            df_sorted = df.sort_values(date_col)
            ax7.plot(df_sorted[date_col], df_sorted[numeric_col])
            ax7.set_title(f'Time Series: {numeric_col} over Time')
            ax7.set_xlabel(date_col)
            ax7.set_ylabel(numeric_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plots.append(fig7)
        
        # 8. Feature Importance Plot (FIXED VERSION)
        try:
            # Check if we have a model with feature importances
            # This assumes you have a trained model in your global scope
            # If not, this will create a correlation-based importance plot
            
            fig8, ax8 = plt.subplots(figsize=(12, 8))
            
            # Method 1: If you have a trained model with feature_importances_
            # Uncomment and modify this based on your actual model
            """
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X.columns
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Take top 15 features
                n_features = min(15, len(importances))
                top_indices = indices[:n_features]
                
                y_pos = np.arange(n_features)
                importance_values = importances[top_indices]
                feature_names_top = [feature_names[i] for i in top_indices]
                
                # FIXED: Proper color handling
                # Option 1: Single color
                colors = 'steelblue'
                
                # Option 2: Gradient based on importance values (uncomment if preferred)
                # norm = plt.Normalize(importance_values.min(), importance_values.max())
                # colors = plt.cm.viridis(norm(importance_values))
                
                # Option 3: List of colors (ensure it matches length)
                # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                #              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                # colors = color_list[:n_features]
                
                ax8.barh(y_pos, importance_values, color=colors)
                ax8.set_yticks(y_pos)
                ax8.set_yticklabels(feature_names_top)
                ax8.invert_yaxis()
                ax8.set_xlabel('Importance Score')
                ax8.set_title('Feature Importances (Model-based)')
            """
            
            # Method 2: Correlation-based feature importance (fallback)
            if len(numeric_cols) > 1:
                # Calculate correlation with target or first numeric column
                target_for_corr = 'target' if 'target' in df.columns else numeric_cols[0]
                other_cols = [col for col in numeric_cols if col != target_for_corr]
                
                if len(other_cols) > 0:
                    correlations = []
                    valid_cols = []
                    
                    for col in other_cols:
                        corr = df[[target_for_corr, col]].dropna().corr().iloc[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                            valid_cols.append(col)
                    
                    if len(correlations) > 0:
                        # Sort by correlation
                        sorted_idx = np.argsort(correlations)[::-1]
                        n_features = min(15, len(sorted_idx))
                        
                        y_pos = np.arange(n_features)
                        importance_values = [correlations[i] for i in sorted_idx[:n_features]]
                        feature_names_top = [valid_cols[i] for i in sorted_idx[:n_features]]
                        
                        # FIXED: Proper color handling
                        # Create a colormap based on importance values
                        norm = plt.Normalize(min(importance_values), max(importance_values))
                        colors = plt.cm.RdYlGn(norm(importance_values))
                        
                        ax8.barh(y_pos, importance_values, color=colors)
                        ax8.set_yticks(y_pos)
                        ax8.set_yticklabels(feature_names_top)
                        ax8.invert_yaxis()
                        ax8.set_xlabel('Absolute Correlation with Target')
                        ax8.set_title(f'Feature Importance (Correlation with {target_for_corr})')
                        
                        # Add colorbar
                        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
                        sm.set_array([])
                        plt.colorbar(sm, ax=ax8, label='Correlation Strength')
                        
                        plt.tight_layout()
                        plots.append(fig8)
            
        except Exception as e:
            print(f"Feature importance plot error (non-critical): {e}")
            # Create a placeholder if feature importance fails
            fig8, ax8 = plt.subplots(figsize=(10, 6))
            ax8.text(0.5, 0.5, 'Feature Importance Plot\nCould not be generated', 
                    ha='center', va='center', transform=ax8.transAxes, 
                    fontsize=14, style='italic')
            ax8.set_title('Feature Importance (Not Available)')
            ax8.axis('off')
            plt.tight_layout()
            plots.append(fig8)
        
        # 9. Density Plots for Numerical Features
        if len(numeric_cols) > 0:
            n_density = min(6, len(numeric_cols))
            fig9, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:n_density]):
                if i < len(axes):
                    df[col].dropna().plot(kind='density', ax=axes[i])
                    axes[i].set_title(f'Density Plot - {col}')
                    axes[i].set_xlabel(col)
            
            # Hide empty subplots
            for i in range(n_density, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Density Plots of Numerical Features', y=1.02)
            plt.tight_layout()
            plots.append(fig9)
        
        # 10. Summary Statistics Table
        if len(numeric_cols) > 0:
            fig10, ax10 = plt.subplots(figsize=(12, 6))
            ax10.axis('tight')
            ax10.axis('off')
            
            # Calculate summary statistics
            summary_stats = df[numeric_cols].describe().round(2)
            
            # Create table
            table = ax10.table(cellText=summary_stats.values,
                              rowLabels=summary_stats.index,
                              colLabels=summary_stats.columns,
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.15] * len(summary_stats.columns))
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            ax10.set_title('Summary Statistics', fontsize=14, pad=20)
            plt.tight_layout()
            plots.append(fig10)
        
    except Exception as e:
        print(f"Error in create_eda_plots: {e}")
        # Create a fallback plot
        fig_error, ax_error = plt.subplots(figsize=(10, 6))
        ax_error.text(0.5, 0.5, f'EDA Plots Error\n{str(e)[:100]}...', 
                     ha='center', va='center', transform=ax_error.transAxes,
                     color='red', fontsize=12)
        ax_error.set_title('Error Generating Plots')
        ax_error.axis('off')
        plots.append(fig_error)
    
    return plots
