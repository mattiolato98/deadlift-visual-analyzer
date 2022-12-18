import pandas as pd


def split_classes(df):
    return df[df['label'] == 'up'], df[df['label'] == 'down']


def drop_column(df, col_name):
    return df.drop([col_name], axis=1)


def remove_header(df):
    df.columns = df.iloc[0]
    return df[1:]


if __name__ == '__main__':
    coords_df = pd.read_csv('coords.csv')
    coords_df = coords_df.drop([f'v{i}' for i in range(33)], axis=1)
    coords_df.insert(0, 'name', [f'{i}'.zfill(6) for i in range(len(coords_df))])

    df_up, df_down = split_classes(coords_df)

    df_up = drop_column(df_up, 'label')
    df_down = drop_column(df_down, 'label')

    df_up = remove_header(df_up)
    df_down = remove_header(df_down)

    df_up.to_csv('deadlift_up.csv', index=False)
    df_down.to_csv('deadlift_down.csv', index=False)
