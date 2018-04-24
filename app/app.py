import numpy as np
import json
import pandas as pd
from flask import Flask, request
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, MinMaxScaler

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')


# From CSV file: default Animal_Data
def load_data(path):
    df = pd.read_csv(path)
    num_instances = df.shape[0]

    new_col = np.arange(num_instances)
    df.insert(loc=0, column='id', value=new_col)

    new_col = np.ones(num_instances)/2.0
    df.insert(loc=2, column='prob', value=new_col)

    new_col = np.zeros(num_instances)
    df.insert(loc=3, column='labeled', value=new_col)

    instances = df
    features = normalize(df.drop('Animal', axis=1).as_matrix())
    return [instances, features, num_instances]


state = {}
# each instance includes three parts: features, name, prob
state['instances'], state['features'], state['num_instances'] = load_data('./data/Animal_Data.csv')
# state['instances'] = instances
# state['features'] = features
similarities = cosine_similarity(state['features'])
# labels = []
state['history'] = []


# transfer matrix to dictionary
def matrix2links(m):
    # m: np matrix
    cols = ['source', 'target', 'value']
    links = []
    for (x, y), value in np.ndenumerate(m):
        if x != y:
            links.append([x, y, value])
    return pd.DataFrame(links, columns=cols)


links = matrix2links(similarities)


@app.route('/')
def homepage():
    return app.send_static_file('index.html')


# Soliciting effective training examples from bootstrap
@app.route('/elements', methods=['GET', 'POST'])
def get_elements():
    return state['instances'].to_csv()


# Soliciting effective training examples from bootstrap
@app.route('/training', methods=['POST'])
def training():

    instances = state['instances']
    labeled_instances = json.loads(request.data)

    history = {'prob': instances[['prob']], 'labeled': instances[['labeled']]}
    state['history'].append(history)

    for labeled_instance in labeled_instances:
        index = labeled_instance['id']
        instances.set_value(index, 'prob', labeled_instance['prob'])
        instances.set_value(index, 'labeled', 1)

    # Update supervised machine learning
    train_data = instances[(instances.labeled != 0)]
    weights = train_data[['prob']].copy()
    y = train_data[['prob']].copy()
    x = train_data.drop('labeled', 1).drop('prob', 1).drop('id', 1).drop('Animal', 1)

    weights = weights.apply(lambda i: abs(i - 0.5) * 2.0)
    weights = weights.values.ravel()
    y = y.apply(lambda i: np.floor(i + 0.5))
    y = y.values.ravel()

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(x, y, sample_weight=weights)

    test_data = instances[(instances.labeled == 0)]
    test = test_data.copy().drop('labeled', 1).drop('prob', 1).drop('id', 1).drop('Animal', 1)
    prob = clf.predict_proba(test)
    test_data = test_data.drop('prob', 1)
    test_data.insert(loc=2, column='prob', value=prob[:, 1])

    # instances = pd.concat([train_data, test_data])
    # instances = instances.sort_index(ascending=True)
    # print(instances)

    # the order of this part.
    state['instances'] = pd.concat([train_data, test_data]).sort_index(ascending=True)


    # print(state['instances'])

    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/undoTraining', methods=['GET'])
def undo_training():
    instances = state['instances']
    history = state['history'].pop()
    instances[['prob']] = history['prob']
    instances[['labeled']] = history['labeled']
    state['instances'] = instances
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


@app.route('/resetTraining', methods=['GET'])
def reset_training():
    instances = state['instances']
    history = state['history'].pop()
    instances[['prob']] = np.ones(state['num_instances'])/2.0
    instances[['labeled']] = np.zeros(state['num_instances'])

    state['instances'] = instances
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


# Soliciting effective training examples from bootstrap
@app.route('/links', methods=['GET', 'POST'])
def get_links():
    return links.to_csv()


# about
@app.route('/about', methods=['GET', 'POST'])
def about():
    return '<h1>Intern Project for Nokia Bell Labs, by Yali Bian</h1>'


# if __name__ == '__main__':
app.run(debug=True)
