Breve documentación de funciones principales utlizadas.
============================================================

obtener_archivos_recursivamente: dada una ruta, obtiene una colección con el nombre de la Query, el nombre del archivo y la ruta en la que se encuentra. Es usada para posteriormente crear un dataframe.

estadistica_exito_queries: devuelve un grafico con la relacion entre queries exitosas/fallidas

get_df_exitosas_nombres: devuelve un dataframe con la estructura ("Query": titulo query, "Data": nombre data query, "Users" contenido data query) para las querys con resultados exitosos.

calcularPondTweet: valora un tweet dándole una puntuación artificial en función de sus estadísticas.

calcularMaxPond: dado un dataframe va a ir ponderando los tweets en él para calcular la máxima ponderación.

maxLikesDataframe: recorre las exitosas y calcula para cada una su tweet más relevante en función de los likes de cada uno.

maxPondDataframe: Función que recorre las exitosas y calcula para cada una su tweet más relevante. La ponderación se evalua en el metodo calcularPondTweet(likes, rt, com, cit)

mostrarViralidadTweets: Dados los tweets más virales de cada consulta, clasifica y visualiza los datos a nivel global del dataframe.Toma como referencia las ponderaciones del tweet más viral de cada consulta.

=============================================================
PROCESAMIENTO DE TEXTO
=============================================================
getTextos: saca los textos del dataframe así como otras características del tweet como el autor, le fecha o el id. Además como segundo parámetro tiene un selector para permitir o no el análisis de retweets utilizando la palabra "no" para evitarlo.

crearDfUsers: devuelve un dataframe de usuarios con la estructura [Id, Nombre] donde son los datos del usuario

traducirIdUser: dado un id, encuentra en el dataframe el nombre correspondiente. Si no se encuentra el id del usuario asignará un guión.

crearDfTextos: creación de un dataframe de textos con la estructura ["Query","Id", "Fecha", "Mes","User", "RespuestaAUser", "Texto"]. Éste será el dataframe principal para el análisis textual y contendrá la información de cada tweet.

crearGrafo: Primer método de creación de grafo que exporta en formato graphml

crearGrafo2: Segundo método de creación de grafo que muestra el grafo utilizando Matplotlib.

tokenize: tokeniza los textos, eliminando palabras que comiencen por @ para evitar menciones.

eliminar_de_querys: elimina los tokens correspondientes a las querys para así tener resultados más limpios y relevantes.

remove_stop: elimina las "stopwords" o palabras vacías.

remove_accent: elimina acentos de los tokens para igualar así todas las palabras.

count_words: Medición de la frecuencia de las palabras ya limpias

wordcloud: Crea una nube de palabras que representa la frecuencia de aparición de términos.

compute_idf: calcula el indice de frecuencia de los términos.

count_keywords: cuenta palabras clave.

count_keywords_by: dado un periodo temporal, cuenta las palabras clave para posteriormente ser comparadas.



