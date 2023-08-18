import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from wordcloud import WordCloud
import regex as re
import numpy as np
import nltk
from collections import Counter
from datetime import datetime
from tqdm import tqdm
from unidecode import unidecode
from nltk.corpus import stopwords
import networkx as nx
import scipy as sp




#Este archivo recoge en un dataset todos los archivos de la colección de tweets

def obtener_archivos_recursivamente(ruta):
    datos_archivos = []
    
    elementos = os.listdir(ruta)
    
    for elemento in elementos:
        elemento_ruta = os.path.join(ruta, elemento)
        
        if os.path.isfile(elemento_ruta):
            datos_archivos.append({
                "Query": elemento_ruta.split("\\")[len(elemento_ruta.split("\\"))-2],
                "Nombre": elemento,
                "Ruta": elemento_ruta

            })
            
        elif os.path.isdir(elemento_ruta):
            archivos_subcarpeta = obtener_archivos_recursivamente(elemento_ruta)
            datos_archivos.extend(archivos_subcarpeta)
    
    return datos_archivos

ruta_raiz = r'C:\Users\Usuario\Desktop\Practicas\datos\tuits_2019\queries'

datos_archivos = obtener_archivos_recursivamente(ruta_raiz)

df_archivos = pd.DataFrame(datos_archivos, columns=["Query","Nombre", "Ruta"])  # Crear el DataFrame

# Acceder a cada fila del DataFrame
#print(df_archivos)


#print(df_archivos.iloc[1]["Ruta"]) # asi accedo a la fila 1 del dataset. Iloc filas


#print(open(df_archivos.iloc[1]["Ruta"]).readlines()[0]) #Asi accedo a la query. 



#Funcion que devuelve el numero total de queries que hay
def contar_num_queries(df):
    q=primera_fila = df.iloc[0]["Query"]
    cont=1
    for indice, fila in df.iterrows():
        if(fila[0]!=q):
            cont+=1
            q=fila[0]
    return cont

#Funcion que devuelve un grafico con la relacion entre queries exitosas/fallidas
def estadistica_exito_queries(df):
    q= [] #queries con resultados satisfactorios
    not_q=[] #queries con resultados no satisfactorios
    cont_d=0
    for indice, fila in df.iterrows():
        
        if(fila[1][:5]=="data_"):
            if(len(fila[1])!=10):
                #print("Datos no vacios "+ fila[1])
                cont_d+=1
                if(fila[0] not in q):q.append(fila[0])
            else: 
                if(fila[0] not in not_q):not_q.append(fila[0])
    
    # Crear los datos para el gráfico circular
    sizes = [len(q), len(not_q)]
    labels = ['Queries exitosas', 'Queries fallidas']
    colors = ['lightgreen', 'lightcoral']

    # Generar el gráfico circular
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Proporciona una forma circular
    plt.title('Distribución de Queries')
    
    # Mostrar el gráfico
    plt.show()



#Funcion que devuelve un dataframe con la estructura ("Query": titulo query, "Data": nombre data query, "Users" contenido data query)
def get_df_exitosas_nombres(df):
    q = []  # queries con resultados satisfactorios
    df_final = pd.DataFrame(columns=["Query", "Data", "Users"])  # DataFrame final

    for indice, fila in df.iterrows():
        if fila[1][:5] == "data_":
            if len(fila[1]) != 10:
                if fila[0] not in q:
                    q.append(fila[0])

    for indice, fila in df.iterrows():
        if fila[0] in q:
            query = fila[0]
            query_arch = fila[1]
            if query in df_final["Query"].values:
                # Buscar si ya existe una entrada con la misma Query
                existing_row_index = df_final.index[df_final["Query"] == query].tolist()[0]
                if df_final.loc[existing_row_index, "Data"] == "" and query_arch != "query":
                    # Comprobar si el valor Data está vacío y asignar fila[1] si no está vacío
                    df_final.loc[existing_row_index, "Data"] = query_arch
                elif df_final.loc[existing_row_index, "Data"] != "" and df_final.loc[existing_row_index, "Users"] == "" and query_arch != "query":
                    df_final.loc[existing_row_index, "Users"] = query_arch
            else:
                # Crear una nueva entrada en el DataFrame
                new_row = {}
                if query_arch[:5] == "data_":
                    new_row = {"Query": query, "Data": query_arch, "Users": ""}
                elif query_arch[:6] == "users_":
                    new_row = {"Query": query, "Data": "", "Users": query_arch}
                df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)

    

    return df_final



df_exitosas=get_df_exitosas_nombres(df_archivos)  #Dataframe formateado con las búsquedas exitosas
"""
# Acceder a los datos de una query
for item in datos:
    public_metrics = item["public_metrics"]
    text = item["text"]
    edit_history_tweet_ids = item["edit_history_tweet_ids"]
    entities = item["entities"]
    id = item["id"]
    created_at = item["created_at"]
    conversation_id = item["conversation_id"]
    lang = item["lang"]
    source = item["source"]
    possibly_sensitive = item["possibly_sensitive"]
    author_id = item["author_id"]
    referenced_tweets = item["referenced_tweets"]
    #geo = item["geo"]

"""
################
#Función que dado un resultado de query
def calcularMaxLikes(datos):
    max_like=datos[0]["public_metrics"]["like_count"]
    max_id=datos[0]["id"]
    for item in datos:
        public_metrics = item["public_metrics"]
        if(max_like<public_metrics["like_count"]): 
            max_like=public_metrics["like_count"]
            max_id=id
    
    return max_like, max_id

def calcularPondTweet(likes, rt, com, cit):
    score=0
    
    score=score+(5*cit)#cada citado cuenta x5
    score=score+(4*rt)#cada rt cuenta x4
    score=score+(3*likes)#cada like cuenta x3
    score=score+(2*com)#cada comentario cuenta x2

    return score


def calcularMaxPond(datos):
    primero=datos[0]["public_metrics"]
    max_pond=calcularPondTweet(primero["like_count"],primero["retweet_count"],primero["reply_count"],primero["quote_count"])
    tx=datos[0]["text"]
    for tweet in datos:
        metricas=tweet["public_metrics"]
        likes = metricas["like_count"]
        rt=metricas["retweet_count"]
        com=metricas["reply_count"]
        cit=metricas["quote_count"]
        pond=calcularPondTweet(likes, rt, com, cit)
        if(max_pond<pond): 
            max_pond=pond
            tx=tweet["text"]
            
   
    return max_pond, tx


###Función que recorre las exitosas y calcula para cada una su tweet más relevante en función de los likes de cada uno
def maxLikesDataframe():
    for index, row in df_exitosas.iterrows():
        ruta=ruta_raiz+'\\'+row["Query"]+'\\'+row["Data"]
        with open(ruta, encoding='utf-8') as archivo:
            datos_q = json.load(archivo)

        mx=calcularMaxLikes(datos_q)
        #print(mx)

###Función que recorre las exitosas y calcula para cada una su tweet más relevante 
###La ponderación se evalua en el metodo calcularPondTweet(likes, rt, com, cit)
def maxPondDataframe():
    max_values = []  # Lista para almacenar los valores máximos

    for index, row in df_exitosas.iterrows():
        ruta = ruta_raiz + '\\' + row["Query"] + '\\' + row["Data"]
        with open(ruta, encoding='utf-8') as archivo:
            datos_q = json.load(archivo)

        mx, texto = calcularMaxPond(datos_q)  # Obtener el valor máximo y el texto asociado
        max_values.append((mx, texto))  # Agregar la tupla (valor máximo, texto) a la lista

    return max_values  # Devolver la lista con los valores máximos



#Dados los tweets más virales de cada consulta, clasifica y visualiza los datos a nivel global del dataframe.
#Toma como referencia las ponderaciones del tweet más viral de cada consulta.
def mostrarViralidadTweets():
    valores=[tupla[0] for tupla in maxPondDataframe()]
    count_irrelevantes = 0
    count_relevantes = 0
    count_virales_relevantes = 0

    for valor in valores:
        if valor < 100:
            count_irrelevantes += 1
        elif valor >= 100 and valor <= 500:
            count_relevantes += 1
        else:
            count_virales_relevantes += 1

    labels = ['Irrelevantes', 'Relevantes', 'Virales']
    labels_ly = ['Irrelevantes <100', 'Relevantes >=100', 'Virales >500']
    sizes = [count_irrelevantes, count_relevantes, count_virales_relevantes]

    colors = ['#FF9999', '#66B2FF', '#99FF99']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.legend(labels_ly, loc='best')
    for i, size in enumerate(sizes):
        plt.text(0.5, 0.5, str(size), color='white', fontsize=12, ha='center', va='center')

    plt.show()

    #################################################################################################################
    #PROCESAMIENTO TEXTO
    #################################################################################################################

#Saca los textos del df

def getTextos(datos, permitir_rt):
    textos_query = []
    for tweet in datos:
        tx = tweet["text"]
        f = tweet["created_at"]
        u = tweet["author_id"]
        i=tweet["id"]
        rt=""

        r_uid=""
        respuesta_a_user = tweet.get("in_reply_to_user_id", None)
        if respuesta_a_user is not None:
            r_uid=respuesta_a_user
        else: r_uid="Vacio"

        if permitir_rt == "no":
            if tx[:2] != "RT":
                textos_query.append((f, tx, u, r_uid,i))
        else:
            textos_query.append((f, tx, u, r_uid,i))
    return textos_query




#Dada fecha devuelve el numero del mes
def getMes(timestamp):
    
        # Parsear la cadena de texto en formato de fecha y hora
        dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Obtener el número del mes (1 a 12)
        month_number = dt.month
        
        return month_number

def getUsers(datos):
    # Acceder a la lista de usuarios
    users = []
    i=""
    u=""
    if 'users' in datos:
        for u in datos['users']:
            i=u["id"]
            n=u["username"]
            users.append((i,n))
    else: users.append(("nc,nc"))
    return users

def crearDfUsers():
    q_exitosas = get_df_exitosas_nombres(df_archivos) # DataFrame inicial con Query, Data, Users
    df_final = pd.DataFrame(columns=["Id", "Nombre"])  # DataFrame final

    for index, row in q_exitosas.iterrows():
        ruta = ruta_raiz + '\\' + row["Query"] + '\\' + row["Users"]
        with open(ruta, encoding='utf-8') as archivo:
            datos_q = json.load(archivo)

        tuplas_id_nombre = getUsers(datos_q)

        for tup in tuplas_id_nombre:
            new_row = {"Id": tup[0], "Nombre": tup[1]}
            df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)

    return df_final


dfu = crearDfUsers()
def traducirIdUser(id):
    
    id_str = str(id)
    resultado = dfu.loc[dfu["Id"] == id_str, "Nombre"].values
    if len(resultado) > 0:
        nombre = resultado[0]
    else:
        nombre = "-"  
    return nombre


def crearDfTextos(permitir_rt):
    q_exitosas=get_df_exitosas_nombres(df_archivos) #DataFrame inicial con Query, Data, Users

    df_final = pd.DataFrame(columns=["Query","Id", "Fecha", "Mes","User", "RespuestaAUser", "Texto"])  # DataFrame final
    recuento_textos=[]
    for index, row in q_exitosas.iterrows():
        ruta = ruta_raiz + '\\' + row["Query"] + '\\' + row["Data"]
        with open(ruta, encoding='utf-8') as archivo:
            datos_q = json.load(archivo)

        
        tuplas_fecha_texto=getTextos(datos_q,permitir_rt)
        query=row["Query"]

        new_row={}
        for tup in tuplas_fecha_texto:
            if tup[1] not in recuento_textos:
                new_row = {"Query": query,"Id":tup[4], "Fecha": tup[0],"Mes":getMes(tup[0]),"User":tup[2],"RespuestaAUser":tup[3], "Texto": tup[1]}
                recuento_textos.append(tup[1])
                #print(len(df_final))
                df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
        

    return df_final

def sacarListaQuerys(df):
    l=[]
    base=df["Query"]
    for item in base:
        sub=item.split()
        for subitem in sub:
            if subitem not in l:l.append(subitem)
        
    return l

def sacarConRespuestasAUsers(df):
    mask = df["RespuestaAUser"] != "Vacio"
    return df[mask]



def crearGrafo(df):
    G = nx.DiGraph()

    # Añadir nodos
    G.add_nodes_from(traducirIdUser(df['User']))

    # Añadir aristas entre los nodos si 'RespuestaAUser' no es nulo
    for index, row in df.iterrows():
        if not pd.isnull(row['RespuestaAUser']):
            G.add_edge(traducirIdUser(row['User']), traducirIdUser(row['RespuestaAUser']))

    # Etiqueta para el grafo
    G.graph['nombre'] = 'Grafo de Tweets'

    # Guardar el grafo en formato GraphML
    nx.write_graphml(G, r'C:\Users\Usuario\Desktop\Practicas\graphv1tweets.graphml')

def crearGrafo2(df):
    G = nx.from_pandas_edgelist(df, source='User', target='RespuestaAUser', create_using=nx.DiGraph())
    sizes = [x[1] * 100 for x in G.degree()]

    # Dibujar el grafo
    pos = nx.spring_layout(G, seed=42) 

    nx.draw_networkx(G, pos, node_size=sizes, with_labels=False, alpha=0.6, width=0.3, node_color='skyblue')

    # Agregar etiquetas a los nodos
    labels = {node: traducirIdUser(node) for node in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', verticalalignment='center')

    plt.axis('off')
    plt.show()


df=crearDfTextos("si");    #dataFrame con estructura ["Query", "Fecha", "Mes", "Texto"]. Permite o no procesar RT


resp=sacarConRespuestasAUsers(df)
crearGrafo2(resp)

token_querys=sacarListaQuerys(df)

#Usuarios con más tweets
grouped_df = df.groupby("User").count()
grouped_df_sorted = grouped_df.sort_values(by="Query", ascending=False)

#######################################################################################################33
def eliminar_de_querys(tokens):
    f=[]
    for t in tokens:
        if(t not in token_querys):f.append(t)
    return f


#Tokeniza, elimina palabras que empiecen por @
def tokenize(text):
    lista_pre_eliminar_querys=re.findall(r'\b(?<!@)\p{L}[\w-]*\b', text)
    lista_fin=eliminar_de_querys(lista_pre_eliminar_querys)

    return lista_fin



# Palabras vacias a eliminar. Carga las iniciales y en función a pruebas cargamos las propias

nltk.download('stopwords')
stopwords_spanish = set(stopwords.words('spanish'))
include_stopwords = {'https','t', 'co', 'sólo', 'rt'}
include_stopwords=set(include_stopwords)
stopwords_combined = stopwords_spanish | include_stopwords

def remove_stop(tokens):
 return [t for t in tokens if t.lower() not in stopwords_combined]

def remove_accent(tokens):
    tokens_sin_acento = [unidecode(token) for token in tokens]
    return tokens_sin_acento
#print(remove_accent({"casa", " cása","féja","eja"}))
#Ahora preparamos el flujo de trabajo 
pipeline = [str.lower, tokenize, remove_stop, remove_accent]
def prepare(text, pipeline):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens

df['tokens'] = df['Texto'].apply(prepare, pipeline=pipeline)
df['num_tokens'] = df['tokens'].map(len)
counter = Counter()
df['tokens'].map(counter.update)

#Medición de la frecuencia de las palabras ya limpias
def count_words(df, column='tokens', preprocess=None, min_freq=2):

    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)

    counter = Counter()
    df[column].map(update)

    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'
    
    return freq_df.sort_values('freq', ascending=False)

freq_df = count_words(df)
freq_df.head(5)

#Gráfico de las 15 palabras más frecuentes
ax = freq_df.head(15).plot(kind='barh', width=0.95)
ax.invert_yaxis()
ax.set(xlabel='Frequency', ylabel='Token', title='Palabras más frecuentes')


#Función para crear bune de palabras
def wordcloud(word_freq, title=None, max_words=200, stopwords=None):

    wc = WordCloud(width=800, height=400, 
                   background_color= "black", colormap="Paired", 
                   max_font_size=150, max_words=max_words)
    
    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq

    if stopwords is not None:
        counter = {token:freq for (token, freq) in counter.items() 
                              if token not in stopwords}
    wc.generate_from_frequencies(counter)
 
    plt.title(title) 

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    

freq_gen_df = count_words(df)
plt.figure(figsize=(12,4))
wordcloud(freq_gen_df['freq'],title="Nube con palabras mas frecuentes", max_words=100)



#Funcion que usa el indice de frecuencia idf (en relacion a la importancia)
def compute_idf(df, column='tokens', preprocess=None, min_df=2):

    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))

    counter = Counter()
    df[column].apply(update)

    idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
    idf_df = idf_df.query('df >= @min_df')
    idf_df['idf'] = np.log(len(df)/idf_df['df'])+0.1
    idf_df.index.name = 'token'
    return idf_df


idf_df=compute_idf(df)
freq_df = freq_df.join(idf_df)

#Comparación enero/octubre
freq_1 = count_words(df[df['Mes'] == 1])
freq_10 = count_words(df[df['Mes'] == 10])

freq_1['tfidf'] = freq_1['freq'] * idf_df['idf']
freq_10['tfidf'] = freq_10['freq'] * idf_df['idf']

plt.figure(figsize=(12,6)) 
plt.subplot(2,2,1)
wordcloud(freq_1['freq'], title='Enero - TF')#usando solo la cuenta en relacion al total
plt.subplot(2,2,2)
wordcloud(freq_10['freq'], title='Octubre - TF'    )
plt.subplot(2,2,3)
wordcloud(freq_1['tfidf'], title='Enero - TF-IDF')#mezclando la cuenta en relacion al total y la idf
plt.subplot(2,2,4)
wordcloud(freq_10['tfidf'], title='Octubre - TF-IDF' )


#5 palabras más frecuentes
keywords = freq_gen_df.index[:5]


#Comparación de datos durante periodo temporal
def count_keywords(tokens, keywords): 
    tokens = [t for t in tokens if t in keywords]
    counter = Counter(tokens)
    return [counter.get(k, 0) for k in keywords]

def count_keywords_by(df, by, keywords, column='tokens'):
    
    df = df.reset_index(drop=True) 
    freq_matrix = df[column].apply(count_keywords, keywords=keywords)
    freq_df = pd.DataFrame.from_records(freq_matrix, columns=keywords)
    freq_df[by] = df[by]
    
    

    return freq_df.groupby(by=by).sum().sort_values(by)

#Vamos viendo cada mes
freq_df = count_keywords_by(df, by='Mes', keywords=keywords)
freq_df.plot(kind='line')
plt.title('Frecuencia de palabras clave por Mes')
#Muestra de los graficos
plt.show()