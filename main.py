import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import iqr
import seaborn as sns

# Creando el arreglo solicitado.
rows = 500
cols = 6
# El valor de arange indica 10001 ya que la función excluye el último número.
df_array = np.array([np.random.choice(np.arange(1, 10001), size=cols,
                                      replace=True) for _ in range(rows)])
print(df_array)

# Convierto en DataFrame el array anteriormente creado para poder asignarle
# nombres a cada columna.
df = pd.DataFrame(df_array)
print(df)

# Cambio los nombres de las columnas

df_final = df.rename(columns={0: 'vegetales', 1: 'leche', 2: 'abarrotes',
                              3: 'congelados', 4: 'limpieza', 5: 'gourmet'})
print(df_final)

# Gráfico de distribución.
graph_distr = sns.displot(df_final, kde=True)
plt.show()

# Gráfico de BoxPlot.
box_plot = sns.boxplot(data=df_final)
plt.show()

# Preprocesamiento por Standard Scaler.
x = df_final.values
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)
df_normalizado = pd.DataFrame(x_scaled)
print(df_normalizado)

# Vuelvo a nombrar las columnas

df_final_normalizado = (df_normalizado.rename(columns=
                                              {0: 'vegetales',
                                               1: 'leche',
                                               2: 'abarrotes',
                                               3: 'congelados',
                                               4: 'limpieza',
                                               5: 'gourmet'}))
print(df_final_normalizado)

describe_df_final_normal = df_final_normalizado.describe()
print(describe_df_final_normal)

IQR = iqr(describe_df_final_normal, axis=0)
print(IQR)

# Parto definiendo los valores del límite inferior.
loc_25 = (describe_df_final_normal.iloc[4, 0:6] - 1.5 * IQR).to_frame()
print(loc_25)

loc_75 = (describe_df_final_normal.iloc[6, 0:6] + 1.5 * IQR).to_frame()
print(loc_75)

# Aplico filtros para sacar los outliers solicitados y luego concateno
df_T_IQ1 = (df_final_normalizado[((df_final_normalizado.vegetales >=
                                   loc_25.iloc[0, 0]) &
                                  (df_final_normalizado.vegetales <=
                                   loc_75.iloc[0, 0]))])
df_T_IQ2 = (df_final_normalizado[((df_final_normalizado.leche >=
                                   loc_25.iloc[1, 0]) &
                                  (df_final_normalizado.leche <=
                                   loc_75.iloc[1, 0]))])
df_T_IQ3 = (df_final_normalizado[((df_final_normalizado.abarrotes >=
                                   loc_25.iloc[2, 0]) &
                                  (df_final_normalizado.abarrotes <=
                                   loc_75.iloc[2, 0]))])
df_T_IQ4 = (df_final_normalizado[((df_final_normalizado.congelados >=
                                   loc_25.iloc[3, 0]) &
                                  (df_final_normalizado.congelados <=
                                   loc_75.iloc[3, 0]))])
df_T_IQ5 = (df_final_normalizado[((df_final_normalizado.limpieza >=
                                   loc_25.iloc[4, 0]) &
                                  (df_final_normalizado.limpieza <=
                                   loc_75.iloc[4, 0]))])
df_T_IQ6 = (df_final_normalizado[((df_final_normalizado.gourmet >=
                                   loc_25.iloc[5, 0]) &
                                  (df_final_normalizado.gourmet <=
                                   loc_75.iloc[5, 0]))])

df_T_IQ1T = df_T_IQ1[['vegetales']]
df_T_IQ2T = df_T_IQ2[['leche']]
df_T_IQ3T = df_T_IQ3[['abarrotes']]
df_T_IQ4T = df_T_IQ4[['congelados']]
df_T_IQ5T = df_T_IQ5[['limpieza']]
df_T_IQ6T = df_T_IQ6[['gourmet']]
df_T_InterQ = (pd.concat([df_T_IQ1T, df_T_IQ2T, df_T_IQ3T, df_T_IQ4T,
                          df_T_IQ5T, df_T_IQ6T], axis=1, ignore_index=True))
df_T_InterQ.reset_index(inplace=True, drop=True)
print(df_T_InterQ)

# Print a describe para ver que el valor mínimo y máximo esté dentro del rango.

df_T_InterQ_describe = df_T_InterQ.describe()
print(df_T_InterQ_describe)

# Aplicando PCA.

X = df_T_InterQ
pca = PCA(n_components=6)
fit1 = pca.fit_transform(X)
print(fit1)

PCA_para_todo = (pd.DataFrame(data=fit1,
                              columns=['vegetales', 'leche', 'abarrotes',
                                       'congelados', 'limpieza', 'gourmet']))
print(PCA_para_todo)

# Varianza explicada para cada dimensión
print('Los ratios de varianza son:', pca.explained_variance_ratio_)

# Reduciendo dimensiones.

X2 = df_T_InterQ
pca2 = PCA(n_components=2)
fit_2 = pca2.fit_transform(X2)

# Ratios de varianza explicada para dimensión.
print(('Los ratios para 2 dimensiones de varianza son:',
       pca2.explained_variance_ratio_))

PCA_para_2 = pd.DataFrame(data=fit_2, columns=['PCA1', 'PCA2'])
print(PCA_para_2)  #

# Graficando

plt.figure(figsize=(8, 6))
plt.scatter(PCA_para_2['PCA1'], PCA_para_2['PCA2'], c=PCA_para_2['PCA2'])
plt.xlabel('Primer componente principal')
plt.ylabel('Segundo componente principal')
plt.show()

# Fuente: https://bit.ly/3lhWq2p

# Los datos a trabajar

X = fit_2  # fit_2 es el array generado luego de la reducción a 2 dimensiones.

for n_clusters in range(2, 7):

    # Creamos un subplot con 1 fila y dos columnas
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)
    # El primer subplot es el gráfico de silhouette
    # El coeficiente puede ir desde el -1, 1 pero en este ejemplo todos están
    # entre 0 y 1
    ax1.set_xlim([0, 1])
    # Esta expresión (n_clusters+1)*10 es para insertar espacios en blanco
    # entre silhouettes
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    # Inicializa el clúster con n_cluster y un generador random
    clusterer = KMeans(n_clusters=n_clusters, max_iter=300, random_state=42)
    cluster_labels = clusterer.fit_predict(X)
    # El silhouette_score entrega el valor promedio para todas las muestras.
    # Esto da una perspectiva de la densidad y separación de la forma de los
    # clústeres.
    silhouette = silhouette_score(X, cluster_labels)
    print("Para n_clusters =", n_clusters,
          "El coeficiente promedio de silhouette es:", silhouette)

    # Calcular el silhouette scores para cada muestra
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Agrega los resultados de silhouette scores para muestras que
        # perteneces al cluster i y los ordena
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Etiqueta los gráficos de silhouette con el número del clúster en el
        # medio
        ax1.text(0, y_lower + 0.5 * size_cluster_i, str(i))

        # Calcula el nuevo y_lower para el siguiente gráfico.
        y_lower = y_upper + 10

    ax1.set_title("El gráfico silhouette para los variados clústeres.")
    ax1.set_xlabel("Valores del coeficiente silhouette")
    ax1.set_ylabel("Etiqueta del clúster")

    # La línea vertical puntuada para el promedio del coeficiente de silhouette
    ax1.axvline(x=silhouette, color="red", linestyle="--")

    ax1.set_yticks([])  # Limpiando el yaxis labels / ticks
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    # Segundo gráfico muestra los dos componentes provenientes de la reducción.
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=50, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Etiquetando los clústeres.
    centers = clusterer.cluster_centers_
    # Dibujando un círculo en el centro de cada clúster.
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("Visualización de los datos clusterizados.")
    ax2.set_xlabel("Primer componente principal")
    ax2.set_ylabel("Segundo componente principal")

    plt.suptitle(
        ("Análisis Silhouette para clustering KMeans para la muestra del df "
         "con número de clusters = %d" % n_clusters),
        fontsize=14, fontweight='bold')

plt.show()

# Fuente: https://bit.ly/3iEnbMB
# Se utiliza la función inverse_transform del módulo sklearn.
df_reverse_PCA6 = scaler.inverse_transform(x_scaled)

# lo transformo en df para mostrarlo de mejor manera.
df_reverse_a_original = (pd.DataFrame(data=df_reverse_PCA6,
                                      columns=['vegetales', 'leche',
                                               'abarrotes', 'congelados',
                                               'limpieza', 'gourmet']))
print('Df revertido es\n\n', df_reverse_a_original)

# Describo el df encontrado para ver similitudes con el df "original"
print('Describe df revertido\n\n', df_reverse_a_original.describe())

# Describo el df_original que es el normalizado y comparo.
print('Describe df original\n\n', df_final.describe())

