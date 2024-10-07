import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Função para carregar e preparar os dados
def load_and_prepare_data(file_paths):
    dataframes = {}
    for name, path in file_paths.items():
        df = pd.read_csv(path)
        df = preprocess_dataframe(df, name)
        dataframes[name] = df
        print(f"Dados carregados para {name}:")
        print(df.head())  # Exibe as primeiras linhas do DataFrame
    return dataframes


# Função para pré-processar cada DataFrame de acordo com a competição
def preprocess_dataframe(df, name):
    if name == "brasileirao":
        df = df[
            [
                "datetime",
                "home_team",
                "home_team_state",
                "away_team",
                "away_team_state",
                "home_goal",
                "away_goal",
                "season",
                "round",
            ]
        ]
    elif name == "brazilian_cup":
        df = df[
            [
                "round",
                "datetime",
                "home_team",
                "away_team",
                "home_goal",
                "away_goal",
                "season",
            ]
        ]
    elif name == "libertadores":
        df = df[
            [
                "datetime",
                "home_team",
                "away_team",
                "home_goal",
                "away_goal",
                "season",
                "stage",
            ]
        ]
    elif name == "br_football":
        df = df[
            [
                "tournament",
                "home",
                "home_goal",
                "away_goal",
                "away",
                "home_corner",
                "away_corner",
                "home_attack",
                "away_attack",
                "home_shots",
                "away_shots",
                "time",
                "date",
                "ht_diff",
                "at_diff",
                "ht_result",
                "at_result",
                "total_corners",
            ]
        ]
        df.rename(columns={"home": "home_team", "away": "away_team"}, inplace=True)

    # Convertendo colunas de gols para numéricas
    df["home_goal"] = pd.to_numeric(df["home_goal"], errors="coerce").fillna(0)
    df["away_goal"] = pd.to_numeric(df["away_goal"], errors="coerce").fillna(0)
    return df


# Função para calcular gols sofridos
def calculate_goals_conceded(df, team):
    return (
        df[df["away_team"] == team]["home_goal"].sum()
        + df[df["home_team"] == team]["away_goal"].sum()
    )


# Função para analisar partidas
def analyze_matches(df, team):
    total_matches = df.shape[0]
    wins = (df["home_goal"] > df["away_goal"]).sum() + (
        df["away_goal"] < df["home_goal"]
    ).sum()
    draws = (df["home_goal"] == df["away_goal"]).sum()
    losses = total_matches - wins - draws
    goals_scored = df["home_goal"].sum() + df["away_goal"].sum()
    total_goals_conceded = calculate_goals_conceded(df, team)

    results = {
        "Total de Partidas": total_matches,
        "Vitórias": wins,
        "Empates": draws,
        "Derrotas": losses,
        "Gols Marcados": goals_scored,
        "Gols Sofridos": total_goals_conceded,
    }
    return pd.DataFrame(results, index=[0])


# Função para plotar gols por temporada
def plot_goals_by_season(goals_by_season, title):
    if not goals_by_season.empty:
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=goals_by_season,
            x="season",
            y="goals_scored",
            label="Gols Marcados",
            marker="o",
            color="blue",
        )
        sns.lineplot(
            data=goals_by_season,
            x="season",
            y="goals_conceded",
            label="Gols Sofridos",
            marker="o",
            color="red",
        )
        plt.title(title)
        plt.xlabel("Temporada")
        plt.ylabel("Total de Gols")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("Nenhum dado disponível para plotagem.")


# Função para plotar distribuição de gols
def plot_goals_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(
        df["home_goal"],
        bins=10,
        kde=True,
        color="blue",
        label="Gols em Casa",
        stat="density",
    )
    sns.histplot(
        df["away_goal"],
        bins=10,
        kde=True,
        color="red",
        label="Gols Fora",
        stat="density",
    )
    plt.title("Distribuição de Gols")
    plt.xlabel("Gols")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    st.pyplot(plt)


# Carregar os datasets
file_paths = {
    "brasileirao": "Brasileirao_Matches.csv",
    "brazilian_cup": "Brazilian_Cup_Matches.csv",
    "libertadores": "Libertadores_Matches.csv",
    "br_football": "BR-Football-Dataset.csv",
}
dataframes = load_and_prepare_data(file_paths)

# Configurar o aplicativo Streamlit
st.title("Análise de Partidas de Futebol")

# Permitir que o usuário escolha a competição
selected_competition = st.selectbox("Escolha uma competição", list(dataframes.keys()))
df = dataframes[selected_competition]

# Permitir que o usuário escolha um time
teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
selected_team = st.selectbox("Escolha um time", teams)


# Analisar e plotar
def analyze_and_plot(df, title, team):
    results = analyze_matches(df, team)
    st.write(f"Resultados da Análise para {title}:")
    st.dataframe(results)

    if "season" in df.columns:
        goals_by_season = (
            df.groupby("season")
            .agg(goals_scored=("home_goal", "sum"), goals_conceded=("away_goal", "sum"))
            .reset_index()
        )
        plot_goals_by_season(goals_by_season, title)
    else:
        st.write(f"Não há dados de temporadas disponíveis para {title}.")

    plot_goals_distribution(df)


# Analisar o time selecionado na competição escolhida
analyze_and_plot(df, selected_competition.capitalize(), selected_team)
