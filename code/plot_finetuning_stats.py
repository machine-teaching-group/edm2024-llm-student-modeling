import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import argparse

# Set environment and styling
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
mpl.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 30})

# Define model mapping
model_mapping = {
    'gpt-3.5-fine-tuned-5-epochs': 'GPT-3.5ft-SS',
    'gpt-3.5-fine-tuned-2-epochs': 'GPT-3.5ft-SS',
    'llama-2-70b-chat-fine-tuned-2-epochs': 'Llama2-70Bft-SS',
    'llama-2-70b-chat-fine-tuned-10-epochs': 'Llama2-70Bft-SS',
}

def process_data(model, ref_task, data_path):
    """
    Process data for the given model and reference task.

    Parameters:
    model (str): The model name.
    ref_task (str): The reference task.
    data_path (str): The path to the data directory.

    Returns:
    train_loss_df (pandas.DataFrame): The DataFrame containing the training loss.
    val_loss_df (pandas.DataFrame): The DataFrame containing the validation loss.
    val_bleu_df (pandas.DataFrame): The DataFrame containing the BLEU scores.
    """
    model_name = data_path.split('/')[-1]
    num_epochs = int(model_name.split('-')[-2])

    if 'llama' in model_name:
        # Llama
        train_loss_file = os.path.join(data_path, 'train_loss.csv')
        val_loss_file = os.path.join(data_path, 'validation_loss.csv')
        val_bleu = os.path.join(data_path, 'validation_BLEU.csv')
        train_loss_df = pd.read_csv(train_loss_file, header=None, names=['step', 'unknown', 'loss'])
        val_loss_df = pd.read_csv(val_loss_file, header=None, names=['step', 'unknown', 'loss'])
        val_bleu_df = pd.read_csv(val_bleu, header=None, names=['step', 'unknown', 'BLEU'])

    elif 'gpt' in model_name:
        # GPT
        train_loss_files = [f.path for f in os.scandir(data_path) if f.is_file() and 'step_metrics' in f.name]
        train_loss_files.sort()
        if len(train_loss_files) > 1:
            data_df = pd.DataFrame()
            for file in train_loss_files:
                tmp_df = pd.read_csv(file, header=0, names=['step', 'train_loss', 'train_accuracy', 'valid_loss', 'valid_mean_token_accuracy'])
                if len(data_df) == 0:
                    data_df = tmp_df
                else:
                    last_step = data_df['step'].iloc[-1]
                    tmp_df['step'] = tmp_df['step'] + last_step
                    data_df = pd.concat([data_df, tmp_df], ignore_index=True)

            data_df.to_csv(os.path.join(data_path, 'merged_loss.csv'), index=False)
            data_df = data_df.astype(float)
            data_df['valid_mean_token_accuracy'] = data_df['valid_mean_token_accuracy'] * 100

        else:
            data_df = pd.read_csv(train_loss_files[0], header=0, names=['step', 'train_loss', 'train_accuracy', 'valid_loss', 'valid_mean_token_accuracy'])
            data_df = data_df.astype(float)
            data_df['valid_mean_token_accuracy'] = data_df['valid_mean_token_accuracy'] * 100

        train_loss_df = data_df[['step', 'train_loss']]
        train_loss_df = train_loss_df.rename(columns={'train_loss': 'loss'})
        val_loss_df = data_df[['step', 'valid_loss']]
        val_loss_df = val_loss_df.rename(columns={'valid_loss': 'loss'})
        val_bleu_df = data_df[['step', 'valid_mean_token_accuracy']]
        val_bleu_df = val_bleu_df.rename(columns={'valid_mean_token_accuracy': 'BLEU'})
        val_loss_df = val_loss_df[val_loss_df['loss'].notna()]
        val_bleu_df = val_bleu_df[val_bleu_df['BLEU'].notna()]

    # Keep only the rows with step appearing in val_loss_df
    train_loss_df = train_loss_df[train_loss_df['step'].isin(val_loss_df['step'])]

    # Convert step to epoch
    last_step = int(float(val_loss_df['step'].iloc[-1]))
    scale_value = last_step/(num_epochs)
    train_loss_df['epoch'] = train_loss_df['step'].apply(lambda x: (x/scale_value))
    val_loss_df['epoch'] = val_loss_df['step'].apply(lambda x: (x/scale_value))
    val_bleu_df['epoch'] = val_bleu_df['step'].apply(lambda x: (x/scale_value))

    return train_loss_df, val_loss_df, val_bleu_df

def plot_data(train_loss_df, val_loss_df, val_bleu_df, axs, i, ref_task, model):
    """
    Plot data for the given DataFrames.

    Parameters:
    train_loss_df (pandas.DataFrame): The DataFrame containing the training loss.
    val_loss_df (pandas.DataFrame): The DataFrame containing the validation loss.
    val_bleu_df (pandas.DataFrame): The DataFrame containing the BLEU scores.
    axs (matplotlib.axes.Axes): The Axes object to plot on.
    i (int): The index of the current plot.
    ref_task (str): The reference task.
    model (str): The model name.
    """
    axs[i].plot(train_loss_df['epoch'], train_loss_df['loss'], label='loss: training', marker='+', markersize=15)
    axs[i].plot(val_loss_df['epoch'], val_loss_df['loss'], label='loss: validation', marker='X', markersize=5)
    axs[i].set_yscale('log')
    axs[i].set_xlabel('epoch')
    if i==0:
        axs[i].set_ylabel('\\textbf{loss (log scale)}', fontsize=27)
        axs[i].set_xticks([0, 1, 2, 3, 4, 5])
    if i==1:
        axs[i].legend(loc='upper center', bbox_to_anchor=(0.3, 1.6), ncol=3)
        axs[i].set_xticks([0, 1, 2])
    axs[i].set_ylim([0.01, 3])
    axs[i].set_title(ref_task.replace('hoc4', '[HoCMaze-4]').replace('hoc18', '[HoCMaze-18]')+ ' ' +model_mapping[model], fontsize=25)

    ax2 = axs[i].twinx()
    ax2.plot(val_bleu_df['epoch'], val_bleu_df['BLEU'], label='BLEU/accuracy: validation', color='green', marker='*', markersize=6)
    if i==3:
        ax2.set_ylabel('\\textbf{BLEU/accuracy}', fontsize=27)
        axs[i].set_xticks([0, 2, 4, 6, 8, 10])
    if i==2:
        ax2.legend(loc='upper center', bbox_to_anchor=(1.1, 1.6), ncol=3)
        axs[i].set_xticks([0, 1, 2])

    ax2.set_ylim([0, 100])
    ax2.set_yticks([20, 40, 60, 80])

def main(args):
    """
    Main function to create the plot.

    Parameters:
    args (argparse.Namespace): The parsed command-line arguments.
    """
    # Create the output path if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Create the figure
    fig, axs = plt.subplots(1, 4, figsize=(20, 8.5))

    # Read the data in fine-tuning-data folder
    i = 0
    for model in list(model_mapping.keys()):
        for ref_task in ['hoc4', 'hoc18']:
            # Get the data path
            data_path = os.path.join('data/finetuning_stats', ref_task, model)
            if not os.path.exists(data_path):
                continue

            # Process the data
            train_loss_df, val_loss_df, val_bleu_df = process_data(model, ref_task, data_path)

            # Plot the data
            plot_data(train_loss_df, val_loss_df, val_bleu_df, axs, i, ref_task, model)

            i += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)

    # Save as pdf
    plt.savefig(os.path.join(args.output_path, 'finetuning_stats.pdf'), format='pdf', dpi=1200)
    print('Saved the plot to: ' + os.path.join(args.output_path, 'finetuning_stats.pdf'))

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/finetuning_stats')
    parser.add_argument('--output_path', type=str, default='outputs/hoc/plots')
    args = parser.parse_args()

    # Call main function
    main(args)
