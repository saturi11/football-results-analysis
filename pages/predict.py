import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os


def get_current_directory():
    """Retorna o diretório atual do arquivo."""
    return os.path.dirname(os.path.abspath(__file__))


def load_data(file_name):
    """Carrega um dataset a partir de um arquivo CSV e trata exceções."""
    file_path = os.path.join(get_current_directory(), "../" + file_name)
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        st.error("O arquivo está vazio.")
        return None
    except pd.errors.ParserError:
        st.error("Erro ao analisar o arquivo.")
        return None


def preprocess_data(df, home_goal_col, away_goal_col):
    """Pré-processa os dados, lidando com valores ausentes e não numéricos."""
    df[home_goal_col] = pd.to_numeric(df[home_goal_col], errors="coerce")
    df[away_goal_col] = pd.to_numeric(df[away_goal_col], errors="coerce")

    # Imputação pela média
    imputer = SimpleImputer(strategy="mean")
    df[[home_goal_col, away_goal_col]] = imputer.fit_transform(
        df[[home_goal_col, away_goal_col]]
    )

    # Remover linhas restantes com NaN
    df = df.dropna(subset=[home_goal_col, away_goal_col])

    # Adicionar características adicionais
    df["goal_difference"] = df[home_goal_col] - df[away_goal_col]
    df["total_goals"] = df[home_goal_col] + df[away_goal_col]

    return df


def train_model(df, home_goal_col, away_goal_col):
    """Treina um modelo de Random Forest e retorna o modelo, acurácia e relatório de classificação."""
    X = df[[home_goal_col, away_goal_col, "goal_difference", "total_goals"]].values
    y = (df[home_goal_col] > df[away_goal_col]).astype(int)

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, model.predict(X_test))
    cross_val_scores = cross_val_score(model, X, y, cv=5)
    classification_rep = classification_report(y_test, model.predict(X_test))

    return model, accuracy, cross_val_scores.mean(), classification_rep


def forecast_results(df, num_games, home_goal_col, away_goal_col):
    """Prevê os resultados com base no número de jogos e retorna os totais de gols e vitórias."""
    total_home_goals = df[home_goal_col].mean() * num_games
    total_away_goals = df[away_goal_col].mean() * num_games
    total_wins = ((df[home_goal_col] > df[away_goal_col]).sum() * num_games) / len(df)

    return total_home_goals, total_away_goals, total_wins


def main():
    """Função principal para execução do aplicativo Streamlit."""
    st.title("Previsão de Resultados de Futebol")

    # Dicionário de arquivos de competições
    competitions_files = {
        "Brasileirão": "Brasileirao_Matches.csv",
        "Copa do Brasil": "Brazilian_Cup_Matches.csv",
        "Libertadores": "Libertadores_Matches.csv",
        "BR Football Dataset": "BR-Football-Dataset.csv",
    }

    competitions = {}
    for name, file in competitions_files.items():
        df = load_data(file)
        if df is not None:
            df = preprocess_data(
                df, home_goal_col="home_goal", away_goal_col="away_goal"
            )
            competitions[name] = df

    selected_competition = st.selectbox(
        "Escolha uma competição", list(competitions.keys())
    )
    df = competitions[selected_competition]

    # Permitir que o usuário escolha um time
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel("K"))
    selected_team = st.selectbox("Escolha um time", teams)

    # Permitir que o usuário escolha o número de jogos
    num_games = st.number_input("Quantos jogos você quer prever?", min_value=1, value=1)

    # Verificar se o DataFrame não é None antes de treinar o modelo
    if df is not None and not df.empty:
        model, accuracy, cross_val_accuracy, classification_rep = train_model(
            df, home_goal_col="home_goal", away_goal_col="away_goal"
        )
        st.write(f"Acurácia do modelo: {accuracy:.2f}")
        st.write(f"Acurácia média da validação cruzada: {cross_val_accuracy:.2f}")

        st.subheader("Relatório de Classificação")
        st.text(classification_rep)

        # Botão para prever
        if st.button("Prever Resultados"):
            total_home_goals, total_away_goals, total_wins = forecast_results(
                df, num_games, home_goal_col="home_goal", away_goal_col="away_goal"
            )
            st.write(f"Total de Gols em Casa Previsto: {total_home_goals:.2f}")
            st.write(f"Total de Gols Fora Previsto: {total_away_goals:.2f}")
            st.write(f"Total de Vitórias Previsto: {total_wins:.2f}")

    else:
        st.error("Nenhum dado disponível para treinar o modelo.")


if __name__ == "__main__":
    main()
