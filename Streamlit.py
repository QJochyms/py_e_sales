import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import joblib
from scipy import stats
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

st.set_page_config(page_title='py-e-sale',
                   layout='centered')

df_initial = pd.read_csv(".\\data\\events.csv")
df_featured = pd.read_csv(".\\data\\featured_events.csv", index_col = 0)
df_final = pd.read_csv(".\\data\\visitor_df.csv")

st.title("PY-E-SALE")
st.subheader(""" **Analyse de la base de données 'events' d'une entreprise de e-commerce**""")
st.write("*par JOCHYMS Quentin, KHOUTHIR Zaabar et GONZALEZ Nicolas*")
st.write("""
# 1 - Présentation du DataFrame
""")
st.subheader('*1.1 - Analyse du jeu de données initial*')

st.write("""
Le dataframe étudié provient de Kaggle et est disponible [ici](https://www.kaggle.com/retailrocket/ecommerce-dataset?select=events.csv) \n
On peut trouver d'autres datasets via ce lien qui seraient succeptible d'etoffer l'analyse ci-après mais ils ne seront pas traités ici.
""")
st.write("Le jeu de donnée initial est le suivant :")
st.write(df_initial.head(10))
st.write("""
Initialement, il présentait""", df_initial.shape[0],"""lignes pour seulement""", df_initial.shape[1],"""colonnes. \n
Nous nous interesserons dans un premier temps à la variable **'event'**
""")

#-----   Création du piechart   -----
event_prop = (df_initial.event.value_counts(normalize=True)).round(2)
fig = px.pie(values =event_prop,
        names=event_prop.index,
        title='Proportions des différents events')
st.write(fig)
st.write("""On s'apperçoit que la distribution de la variable **'event'** est très déséquilibrée
 puisque l'occurence *view* est représentée à plus de 95%.""")
#------------------------------------

#-----   Création du barchart   -----
nb_visiteur = df_initial.visitorid.value_counts()

fig, ax = plt.subplots()

plt.xlabel("Nombre d'events créés",fontdict = {'fontsize' : 5})
plt.ylabel("Nombre de visitorid",fontdict = {'fontsize' : 5})
plt.title(label = "Nombre d'events créés par les différents visitorid",fontdict = {'fontsize': 10})
ax.hist(nb_visiteur.values, bins=30, log=True,color='green')
st.write(fig)
st.write("""Au même titre que la variable **'event'**, on remarque ici un déséquilibre
dans la représentation des différents **'customerid'**. \n
 En effet, plus de 100.000 clients ne créent qu'entre 1 et 250 events. \n
 A contrario, trois clients créent plus de 3.000; 4.000 et 7.500 events chacuns.""")
#------------------------------------

st.write(pd.crosstab(df_initial['event'], df_initial['transactionid'].notnull()))
st.write("""Cette table permet d'affirmer que chaque event 'transaction' correspond à un **'transactionid'**. \n
De plus lorsque **'event'** = *view* ou *addtocart*, **'transactionid'** est un NA. \n
Notre jeu de données semble donc propre.""")

#-----   Création du countplot   -----
fig, ax = plt.subplots()
item_per_transaction = df_initial[df_initial['transactionid'].notnull()].groupby(['transactionid']).count()['timestamp']
ax = sns.countplot(item_per_transaction, log = True)
plt.xlabel("Nombre de produits")
plt.ylabel("Nombre de transactions")
plt.title("Nombre d'items par transactions",
         fontdict = {'fontsize': 10})
st.write(fig)
st.write("""Ce graphique nous apporte l'information qu'une transaction (et donc un **'transactionid'**) peut concerner plusieurs produits. \n""")
#------------------------------------

st.write("""Nous allons maintenant modifier le jeu de données pour en exploiter la colonne **'timestamp'** qui est actuellement exprimée en ms et donc inutilisable.""")

st.subheader('*1.2 - Modifications apportées au jeu de données*')
st.write("""Nous apportons un certain nombre de modifications à notre jeu de données pour pouvoir calculer des statistiques
sur la maille **'visitorid'**. \n
Sa version intermédiaire est la suivante :""")

st.write(df_featured.head(10))

st.write("""On y retrouve nos variables initiales (timestamp, visitorid,event,itemid et transactionid). \n
Les variables addtocart; transaction; view ont été obtenues via un pandas.get_dummies sur la variable *event*. \n
Les variables date; month; day; hour; weekday; month-day ont été obtenues à partir de la variable *timestamp* sur laquelle
nous avons appliqué la fonction to_datetime de pandas. \n
La variable diftime(m) correspond à la durée (en minutes) entre deux évènements pour un même utilisateur. Elle vaut NaN
lorsque l'utilisateur n'a réalisé qu'une seule action sur le site. Cette colonne a été créée en utilisant les fonctions 
*.diff()* et *.shift(-1)*. \n
Les variables hourclass correspondent à une segmentation des events en 6 plages horaires de 4 heures chacunes.""")
st.write("""Le fait d'avoir scindé notre colonne **'timestamp'** nous permet d'afficher des données temporelles plus exploitables. \n"""
         """Cela nous permet, entres autres, de nous aperçevoir que les données ont été récoltées entre le """,df_featured["month-day"].min(),
         """et le """, df_featured["month-day"].max())
st.write("Nous allons maintenant réaliser des test de chi2 pour analyser la corrélation entre la variable **'event'** et"
         " nos nouvelles variables temporelles.")
index = ("chi2","p-value","dof","expected")
st.markdown('*Test chi2 event/hour*')
event_hour = pd.crosstab(df_featured['event'], df_featured['hour']).round(2)
st.dataframe(pd.DataFrame(stats.chi2_contingency(event_hour),index = index))
df_chi2_hour = pd.DataFrame(data = stats.chi2_contingency(event_hour))
st.success("""L'hypothèse H0 de non indépendance est rejetée car le chi2 s'établit à """ + str(df_chi2_hour.iloc[0,0].round(2))+ """\n
Ce rejet est corroboré par une p-value à """ + str(df_chi2_hour.iloc[1,0].round(2)))

st.markdown('*Test chi2 event/day*')
event_day = pd.crosstab(df_featured['event'], df_featured['day'])
st.dataframe(pd.DataFrame(stats.chi2_contingency(event_day),index = index))
df_chi2_day = pd.DataFrame(data = stats.chi2_contingency(event_day))
st.success("""L'hypothèse H0 de non indépendance est rejetée car le chi2 s'établit à """ + str(df_chi2_day.iloc[0,0].round(2)))

st.markdown('*Test chi2 event/month*')
event_month = pd.crosstab(df_featured['event'], df_featured['month'])
st.dataframe(pd.DataFrame(stats.chi2_contingency(event_month),index = index))
df_chi2_month = pd.DataFrame(data = stats.chi2_contingency(event_month))
st.success("""L'hypothèse H0 de non indépendance est rejetée car le chi2 s'établit à """ + str(df_chi2_month.iloc[0,0].round(2)))

st.write("On observe donc une bonne corrélation entre les variables **'event'** et **'[hour;day;month]'**. \n"
         "Cependant, plus la maille temporelle est grande, plus le résultat de chi2 augmente et la p-value diminue. \n"
         "La corrélation est donc plus importante pour une maille faible (ici la variable **'hour'**).")
st.write("Nous pouvons également rendre compte de cette corrélation en voyant les graphiques suivant :")

#-----   Création du barplot   -----
cols = ['month-day', 'month', 'weekday','hour']
fig, axes = plt.subplots(len(cols),1, figsize=(25,32))
for col , ax in zip(cols, axes.flat):
    event_agg = df_featured[[col,'view','addtocart','transaction']].groupby(col).sum()
    event_agg.plot(kind='bar', stacked=True, ax=ax)
st.write(fig)
st.write("""
Au vu des graphiques obtenus, on observe une nette corrélation entre notre variable **'event'** et **'timestamp'** et
surtout au niveau des heures. On peut déceler des plages horaires succeptibles d'affecter la création d'event ou non.
On peut également en déduire que les données ont été enregistrées dans un pays avec un fuseau horaire différent du notre. \n
A contrario, les tendances sur les variables **'day'** et **'month'** sont bien moins évidentes, d'autant que le mois de
septembre est tronqué car le dataset s'arrête au""", df_featured["month-day"].max())
#-----------------------------------

st.write("**Après analyse du potentiel du jeu de donnée, nous avons décidé de créer un modèle de classification afin "
         "de classer les visiteurs en fonction de leur potentiel à acheter un produit.**")
st.write("Nous avons donc apporté un certain nombre de modifications à notre jeu de données pour en arriver à sa version finale"
         " qui nous servira de base pour entraîner notre modèle. Il se concentre sur la maille **'customerid'**. \n"
         "La version finale du jeu de donnée est donc la suivante :")

st.write(df_final.head(10))
colonnes = df_final.columns
definition = ("Identifiant du visiteur",
               "Nombre d'evenements du visiteur",
               "Nombre de produits différents consultés par le visiteur",
               "Nombre de 'view'",
               "Nombre de 'add_to_cart'",
               "Nombre de sessions enregistrées",
               "Nombre d'evenements générés par le produit le plus consulté pour le visiteur concerné",
               "Durée de session moyenne",
               "Écart-type durée de session",
               "Durée totale cumulée des sessions",
               "Durée moyenne par 'view'",
               "Durée moyenne par 'add_to_cart'",
               "Nombre d'event sur la période ]00-4] heure",
               "Nombre d'event sur la période ]4-8] heure",
               "Nombre d'event sur la période ]8-12] heure",
               "Nombre d'event sur la période ]12-16] heure",
               "Nombre d'event sur la période ]16-20] heure",
               "Nombre d'event sur la période ]20-00] heure",
               "Nombre d'event intervenus durant un week-end",
               "Indique si le visiteur a déjà acheté sur le site (1) ou non (0)")
description = pd.DataFrame(data = definition,
                           index = colonnes,
                           columns = ["Description variable"])
st.write(description)

st.write("Ce nouveau jeu de données dispose maintenant de ", df_final.shape[0],"lignes pour ", df_final.shape[1],"colonnes. \n"
         "Notre variable cible est **'has_bought'**. \n"
         "Toutes les autres colonnes seront utilisées comme variables explicatives.")

st.subheader('*1.3 - Analyse du DataFrame final*')

st.write("Nous allons maintenant analyser les coefficients de corrélation de nos variables entre elles en utilisant la fonction heatmap de seaborn.")
fig, ax = plt.subplots(figsize=(40,15))
type = st.radio('Choisir le type de corrélation à afficher :', ['spearman','pearson'])
sns.heatmap(data = df_final.corr(method = type), annot=True, cmap='coolwarm', ax = ax)
ax.set_title("Matrice de corrélation - coefficients " + str(type))
st.pyplot(fig)

st.write("Parmis les variables, celles qui possèdent le plus de corrélation avec la variable cible sont: **nb_addtocart** et **mean_time_per_addtocart**.\n" 
         "La variable **nb_event_of_most_interesting_item** semble également porter une part non négligeable d'information.\n"
         "La variable **nb_event** devra etre supprimé car cette variable en combinaison avec **nb_view** et **nb_addtocart** permet de calculer la variable **nb_transaction**.\n"
         "La variable **nb_unique_item** semble très correlée à la variable **nb_view**, celle-ci pourra être supprimer pour garder uniquement **nb_view**.\n"
         "Enfin les variables relatives au temps sont fortement correlées les unes avec les autres. On pourra eliminer les variables **full_time_session** et **mean_time_per_view**.\n"
         "Les résultats ne sont pas vraiment concluants, peu importe la métrique utilisée. La variable cible ne présente pas de corrélation ")

st.write("""
Après analyse des résultats obtenus, d'après la méthode **Elbow** (ou du coude), le nombre optimal de clusters serait 
compris entre 30 et 40. \n"""
"""En effet, à partir de ces valeurs, la variance ne se réduit plus significativement.\n
Nous choisirons ici de retenir la valeur 30.\n
Nous poursuivrons l'analyse des clusters en utilisant un dendrograme via le code suivant :""")

st.code("""
Z = linkage(clusters_df, method='ward', metric='euclidean')
fig, axes = plt.subplots(2,1,figsize=(30,20))

t=8000

dendrogram(Z, labels = clusters_df.index, leaf_rotation = 0, color_threshold = t,ax=axes[0])
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].axhline(y=t)

dendrogram(Z, labels = clusters_df.index, leaf_rotation = 0, color_threshold = t,ax=axes[1]);
axes[1].tick_params(axis='both', which='major', labelsize=20)
axes[1].axhline(y=t)
axes[1].set_yscale('symlog')
""")

st.write("""
Nous obtenons alors les dendrogrammes suivant, le 1er en echelle non logarithmique, le second en echelle logarithmique :""")
graphe2 = Image.open('.\\dendrogramme.PNG')
st.image(graphe2)
st.write("""
Les clusters 6,17,5,27,4,33,1,18,14,2,7 regroupent les visiteurs n'ayant réalisé qu'une action avec très peu de mauvaise
 classification. \n
Les groupes qui regroupes les visiteurs ayant un temps de session quasi exclusivement supérieur à 800 minutes sont: 
31,37,39,29,35,15,16,13,34 qui correspondent au regroupement effectué par la CAH. \n
Quand on regarde le nombre d'achats dans chaque classe, on peut séparer ses visiteurs de la même manière que la cah: \n
- Les clusters 31,37,39,29,35 correspondent à des temps de sessions longues mais presque pas d'achat.
- Les clusters 15,16,13,34 correspondent à des temps de sessions longs mais avec plus de potentiel d'achat.
""")
st.write("""
# 2 - Modélisation
""")
st.subheader('*2.1 - Étude des modèles*')

st.write("Nous avons remarqué que l'entrainement des différents modèles sur nos données est relativement long lorsqu'il est réalisé sur "
         "le dataset complet. "
         "De plus, du fait du caractère deséquilibré du jeu de données, l'apprentissage semble biaisé.. "
         "Afin d'eviter ce problème nous avons essayé de reduire le dataset et notamment d'enlever tout les visiteurs n'ayant réalisé "
         "qu'une seule opération. En effet, nous avons remarqué que la proportion de visiteur passés une fois sur le site"
         "et ayant a acheté est négligeable et donc peu pertinente pour le temps de calcul qu'elle induit.\n"
         "Au total **1.001.591** visiteurs n'ont réalisé qu'une seule visite et **70** d'entre eux ont réalisé un achat sur le site. \n"
         "Nous allons donc réduire le dataset afin de ne conserver que les visiteurs ayant déjà réalisé une action **'addtocart'**.")

#-------------Modification du dataset, à modifier si on veut mettre un choix utilisateur--------------------
event = df_final[df_final['nb_event']!=1]
event=event.drop(['nb_event','full_time_session','mean_time_per_view', 'nb_unique_item'], axis=1)

#-------------Préprocessing du model------------------------------------------------------------------------
target = event['has_bought']
features = event.drop('has_bought', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state = 42)
#-------------Standardisation des données-------------------------------------------------------------------
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Nous avons testé différents modèles via la boucle suivante :")
st.code("""random_state = 42
models = []
models.append(('LR', LogisticRegression(random_state=random_state)))
models.append(('KNN', KNeighborsClassifier(n_jobs = -1)))
models.append(('DTC', DecisionTreeClassifier(max_depth = 10)))
models.append(('RandomForest', RandomForestClassifier(n_jobs = -1)))
models.append(('SVM', SVC(random_state=random_state))) 
models.append(('AdaBoost',AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)
                                            ,random_state=random_state,learning_rate=0.1)))
models.append(('GradientBoost',GradientBoostingClassifier(random_state=random_state)))

# Test des modèles + affichage des résultats
for name, model in models:
    t0 = time()
    prediction = model.fit(X_train_scaled, y_train)
    y_pred = prediction.predict(X_test_scaled)
    matrix = pd.crosstab(y_test, y_pred, rownames = ['Classe réelle'], colnames = ['Classe prédite'])
    elements = classification_report(y_test, y_pred, output_dict = True)
    rapport = pd.DataFrame.from_dict(elements)
    t1 = time() - t0
    msg = ------------------------------------------------------------------------------------------------------------\n
    msg = "Modèle : %s % (name)
    msg = msg +
    print(msg)
    print("Rapport de classification :, rapport,)
    print("Matrice de confusion : , matrix, )
    print("Réalisé en {} secondes".format(round(t1,3)))
    msg_1 = ---------------------------------------------------------------------------------------------------------\n
    print(msg_1)""")
with st.beta_expander("Afficher l'output"):
    st.code(
    """Modèle : LR 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.802691     0.896885  0.814314     0.849788      0.829408
    recall        0.982239     0.390187  0.814314     0.686213      0.814314
    f1-score      0.883435     0.543797  0.814314     0.713616      0.787103
    support    5405.000000  2140.000000  0.814314  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            5309     96
    True             1305    835 
    
    Réalisé en 0.321 secondes
    ------------------------------------------------------------------------------------------------------------
    
    Modèle : KNN 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.833274     0.615150  0.776408     0.724212      0.771407
    recall        0.859944     0.565421  0.776408     0.712683      0.776408
    f1-score      0.846399     0.589238  0.776408     0.717818      0.773460
    support    5405.000000  2140.000000  0.776408  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            4648    757
    True              930   1210 
    
    Réalisé en 3.95 secondes
    ------------------------------------------------------------------------------------------------------------
    
    Modèle : DTC 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.895830     0.748335  0.854738     0.822082      0.853995
    recall        0.902128     0.735047  0.854738     0.818587      0.854738
    f1-score      0.898968     0.741631  0.854738     0.820299      0.854342
    support    5405.000000  2140.000000  0.854738  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            4876    529
    True              567   1573 
    
    Réalisé en 0.406 secondes
    ------------------------------------------------------------------------------------------------------------
    
    Modèle : RandomForest 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.909207     0.800878  0.879788     0.855043      0.878481
    recall        0.924514     0.766822  0.879788     0.845668      0.879788
    f1-score      0.916797     0.783481  0.879788     0.850139      0.878984
    support    5405.000000  2140.000000  0.879788  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            4997    408
    True              499   1641 
    
    Réalisé en 2.441 secondes
    ------------------------------------------------------------------------------------------------------------
    
    Modèle : SVM 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.784175     0.703527  0.772962     0.743851      0.761301
    recall        0.942461     0.344860  0.772962     0.643660      0.772962
    f1-score      0.856063     0.462841  0.772962     0.659452      0.744532
    support    5405.000000  2140.000000  0.772962  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            5094    311
    True             1402    738 
    
    Réalisé en 55.756 secondes
    ------------------------------------------------------------------------------------------------------------
    
    Modèle : AdaBoost 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.889707     0.728950  0.844665     0.809329      0.844111
    recall        0.893987     0.720093  0.844665     0.807040      0.844665
    f1-score      0.891842     0.724495  0.844665     0.808168      0.844377
    support    5405.000000  2140.000000  0.844665  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            4832    573
    True              599   1541 
    
    Réalisé en 0.691 secondes
    ------------------------------------------------------------------------------------------------------------
    
    Modèle : GradientBoost 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.900988     0.768381  0.864414     0.834684      0.863377
    recall        0.910823     0.747196  0.864414     0.829010      0.864414
    f1-score      0.905879     0.757640  0.864414     0.831760      0.863834
    support    5405.000000  2140.000000  0.864414  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            4923    482
    True              541   1599 
    
    Réalisé en 13.762 secondes
    ------------------------------------------------------------------------------------------------------------
    """)

st.subheader('*2.2 - Choix du modèle*')

st.write("""
*     2.2.1 - Choix de la metrique de score

Du fait de notre jeu de données deséquilibré, un score d'accuracy n'aurait pas de sens. En effet, le nombre d'individus 
de la classe 0 étant majoritaire, ce chiffre risque d'induire un score d'accuracy toujours élevé même si la classe 1 
n'est pas bien prédite. \n
L'objectif est de détecter les cibles suceptibles d'acheter un produit afin de pouvoir agir en conséquence 
(proposer une promotion, offrir des frais de port, ou au contraire ne rien faire si les chances sont élevées...).
Notre objectif est donc de maximiser le nombre de classe 1. \n
On s'intéressera donc au rappel. \n
Il parait plus pertinent de ne pas rater d'acheteurs potentiels quitte à inclure des clients non intéressés plutot que 
de passer à côté d'acheteurs potentiels. \n
A ce titre, on s'intéressera plutot au rappel sur la classe 1, pour affiner notre choix, on pourra se reporter au 
f1_score qui prend également en compte la précision pour limiter le nombre de faux positifs.

*     2.2.2 - Choix du modèle de prédiction

Au regard des critères retenus évoqués ci-dessus, le modèle qui donne les meilleurs résulats est le **Random Forest.**
Nous choisirons donc ce modèle pour réaliser l'optimisation des hyperparametre.
Pour rappel, le Random Forest produit les résultats suivants :""")
st.code("""
Modèle : RandomForest 
    
    
    Rapport de classification : 
    
                      False         True  accuracy    macro avg  weighted avg
    precision     0.909207     0.800878  0.879788     0.855043      0.878481
    recall        0.924514     0.766822  0.879788     0.845668      0.879788
    f1-score      0.916797     0.783481  0.879788     0.850139      0.878984
    support    5405.000000  2140.000000  0.879788  7545.000000   7545.000000 
    
    Matrice de confusion : 
    
     Classe prédite  False  True 
    Classe réelle               
    False            4997    408
    True              499   1641 
    """)

st.write("""La courbe ROC ci-après permet d'évaluer la performance de notre modèle. \n
Ce type d'évaluation est adaptée lorsque le jeu de donnée utilisé est déséquilibré, ce qui est notre cas ici.
En effet, lorsque les classes sont très déséquilibrées, la matrice de confusion et surtout le taux d’erreur, donnent 
souvent une fausse idée de la qualité de l’apprentissage. \n
L'aire sous la courbe (Area Under Curve ou AUC) est l'indicateur principal de la courbe ROC. Plus sa valeur est
 proche de 1, plus le modèle est réputé performant. \n
 Par convention, la courbe du modèle est mise en parallèle avec une courbe d'AUC = 0.5. C'est le résultat qui
 serait obtenu par un classifieur aléatoire et donc inutile.""")
#-------------Paramétrage + affichage courbe ROC------------------------------------------------------------
random_state = 42

models = RandomForestClassifier(n_estimators=100,
        random_state=random_state,
        max_features="auto",
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1)

prediction = models.fit(X_train_scaled, y_train)
y_pred = prediction.predict(X_test_scaled)
probs = models.predict_proba(X_test_scaled)

fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label = 1)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10,5))
plt.plot(fpr, tpr, color = 'orange',
         lw = 2, label = 'Modèle clf (auc = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--', label = 'Aléatoire (auc = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux faux positifs')
plt.ylabel('Taux vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc = "lower right")
st.pyplot(fig)

st.write("La courbe ROC obtenue pour notre classifieur Random Forest est très encourageante, on constate une AUC à 0.96."
         "Neanmoins, cela est en partie dû au déséquilibre des données, en effet, la majeure partie des éléments à "
         "prédire appartient à la classe 0, il faut donc se méfier des résultats obtenus.")

st.subheader('*2.3 - Optimisation du modèle*')

st.write("Maintenant que nous avons choisi quel modèle de classification exploiter, nous devons l'optimiser pour le "
         "rendre le plus pertinent et efficace possible au regard de l'objectif formulé. \n"
         "Nous allons ainsi réaliser une batterie de tests pour en arriver à la méthode la plus adaptée. \n"
         "Dans un premier temps, nous allons réduire le dataset en enlevant les clients n'ayant produit qu'un seul"
         "évenement, en effet, cette catégorie d'acheteurs est difficile à prédire et concerne 1.001.591 utilisateurs"
         "sur les 1.407.580 recensés."
         "Il nous reste donc 405.989 utilisateurs à étudier.")

st.write("Ensuite, pour compenser le déséquilibre important de la variable cible, nous allons passer par une technique"
         "de sous-échantillonnage,le **RandomUnderSampler** de la librairie imblearn. Cette classe nous permet d'effectuer "
         "un sous-échantillonnage aléatoire.")
st.code("""from imblearn.under_sampling import RandomUnderSampler
rUs= RandomUnderSampler()
X_ru, y_ru=rUs.fit_resample(X_train_scaled, y_train)
print("samples rus",dict(pd.Series(y_ru).value_counts()))""")

st.write("""
A partir de là, nous avons cherché à optimiser les paramètres de notre classifieur RandomForest.
Pour cela, nous avons utilisé la fonction GridSearchCV sur les hyperparamètres suivants :\n
* n_estimators : [200, 400, 600, 800, 1000]
* criterion : ['gini','entropy']
* max_features : ['sqrt', 'log2']\n
En affichant les *f1_score, f1_macro* et le *recall*. \n
Il en est ressorti que le couple d'hyperparamètres maximisant ces scores sont :\n
* n_estimators : 600
* criterion : 'entropy'
* max_feature : 'sqrt'\n
Nous avons donc entraîné notre modèle avec ces paramètres.""")

st.write("""
L'utilisation de ces hyperparamètres nous a permis d'obtenir le rapport de classification suivant :""")
st.code("""
                   pre       rec       spe        f1       geo       iba       sup

      False       1.00      0.96      0.96      0.98      0.96      0.92     78854
       True       0.44      0.96      0.96      0.61      0.96      0.92      2344

avg / total       0.98      0.96      0.96      0.97      0.96      0.92     81198

Classe prédite	False	True
Classe réelle		
        False   76045	2809
        True      103	2241
""")

st.write("""
Ces résultats sont encourageants car le nombre de faux négatifs (103) est assez faible.
Le rappel de la classe positive est affiché à 96%, ce qui est également très positif.
""")

st.write("""
Nous allons maintenant essayer une autre méthode d'optimisation, le BalancedRandomForestClassifier de imblearn.
En utilisant les même paramètres que précédemment, les résultats obtenus sont les suivants :
""")

st.code("""
                  pre       rec       spe        f1       geo       iba       sup

      False       1.00      0.97      0.96      0.98      0.96      0.93     78854
       True       0.46      0.96      0.97      0.62      0.96      0.93      2344

avg / total       0.98      0.97      0.96      0.97      0.96      0.93     81198

Classe prédite   False	 True
Classe réelle		
        False	 76161	 2693
        True	   94	 2250
""")

st.write("""
Les deux approches sont équivalentes en terme de performances: il y a une différence de 0.02 points sur le score de 
précision par rapport à la classe 1 en utilisant le modèle basé sur 'BalancedRandomForest'.
""")

st.write("""
Ces approches nous permettent d'afficher la liste des features utilisées ainsi que leur importance dans la construction
du modèle.""")

#------------Paramétrage-------------------

#event=event.drop(['nb_event',
#                  'full_time_session',
#                  'mean_time_per_view',
#                  'nb_unique_item',], axis=1)

# Standardisation des données
#rUs = joblib.load('C:\\Users\\nicog\\OneDrive\\Documents\\DataScientest\\Projet\\Rendus\\Streamlit\\Saves\\rf_rus.sav')
#rUs= RandomUnderSampler()
#X_ru, y_ru=rUs.fit_resample(X_train_scaled, y_train)
#rf = RandomForestClassifier()

#rf_balanced = joblib.load('C:\\Users\\nicog\\OneDrive\\Documents\\DataScientest\\Projet\\Rendus\\Streamlit\\Saves\\rf_balanced.sav')
rf_balanced = BalancedRandomForestClassifier(n_estimators=600, max_features='sqrt', criterion='entropy')
rf_balanced.fit(X_train_scaled, y_train)
#y_pred = rf_balanced.predict(X_test_scaled)

feat = pd.Series(rf_balanced.feature_importances_, index=features.columns)
feat = feat.sort_values(ascending=False)
fig, ax = plt.subplots()
ax.barh(y = feat.index, width=feat)
st.pyplot(fig)

st.write("""
# 3 - Résultats
""")

st.write("""
A partir de ces modèles, il est possible de calculer les probabilités d'appartenir à chaque classe: \n

* Soit pour affiner le rapport entre la précision et le rappel et trouver un équilibre satisfaisant pour l'utilisateur du modèle
* Soit pour différencier les clients en fonction de leur probabilité d'acheter. Avec une probabilité très forte ou très 
faible (seuils à définir avec l'utilisateur du modèle), il n'est peut-être pas nécessaire d'avoir une action car 
celle-ci aura peu d'impact.
""")

st.write("""
Nous allons maintenant analyser le couple *precision/rappel* pour les différents seuils de probabilité via la fonction
plot_precision_recall_curve de sklearn.
""")

st.code("plot_precision_recall_curve(rf_balanced, X_test_scaled, y_test)")
st.write("""
L'utilisation de cette fonction produit le graphique suivant :
""")
graphe3 = Image.open('.\\precision_recall.PNG')
st.image(graphe3)

probs = rf_balanced.predict_proba(X_test_scaled)

t = st.slider("Choix de t (seuil d'appartenance) :", min_value = 0.00, max_value = 1.00,step = 0.01, value = 0.85)

st.write("""
Matrice de confusion : \n""",
confusion_matrix(y_test, probs[:,1]>t),'\n',
""" précision : """,precision_score(y_test, probs[:,1]>t).round(4),"""\n""",
""" rappel : """,recall_score(y_test, probs[:,1]>t).round(4)
)

st.write("""
# 4 - Conclusion et prolongements possibles
""")
st.write("""
    Notre modèle de prédiction ne fonctionne pas parfaitement. Mais nous avons malgré tout tenté de
 minimiser le taux de faux négatifs et de maximiser le taux de vrais positifs. Une compréhension plus fine de la 
 manière dont les données sont acquises pourrait aider au choix des variables et peut-être à un meilleur modèle. \n
Nous avons également réalisé une classification (classification non supervisé, modèles des kmeans suivie d’une CAH) des 
visiteurs en fonctions de leurs caractéristiques. Nous nous sommes rendu compte qu’il y avait des classes très 
différentes de visiteurs. Nous aurions pu essayer de comprendre les comportements représentant ces classes. 
Ce découpage du jeu de données aurait pu nous amener à traiter chaque groupe de manière différentes et proposer une 
tache en fonction du groupe : \n
* Pour les visiteurs éphémères : trouver un moyen de les accrocher
* Pour les visiteurs plus intéressés : actions pour améliorer la transformation
* Pour les visiteurs avec un comportement étrange (type robot) : comprendre leur rôle (prospection commerciale ou
 benchmark) et adapté une réponse (affichage de prix sous la concurrence…) \n
Il aurait également été possible d'affiner notre analyse en étudiant les produits consultés (proposer des produits 
différents à un client qui est sur le point de quitter le site, analyse du potentiel d'achat d'un client sur un 
produit/classe de produit) ...
""")