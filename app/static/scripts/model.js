/**
 * Created by Yali on 2/9/17.
 */

// Updated model for calculating document weight and links, and entity weights.

// a weighting vector is applied to each dimension for creating
// Every time the weighting vector changed, changing the M and K


// Once the updated weight vector is computed, the model updates the spring strengths and document masses,
// and the layout iterates until settling again.
var Model;

(function () {

    var data,
        documents,
        edges,
        modelType,
        entities,
        K = 0.1, // Constant for update entity weight
        searches;

    // init entityWeightVector;

    Model = function (x) {

        data = x;
        documents = data.documents;
        edges = data.edges;
        entities = data.entities;
        initModel();
        var model = {};

        // real function used to update Mass and Spring K
        model.data = function (x) {
            if (!arguments.length) {
                return data;
            }
            data = x;
            documents = data.documents;
            edges = data.edges;
            entities = data.entities;

            return model;
        };

        // Changing DIMENSION REDUCTION MODELS FOR SEMANTIC INTERACTION
        model.modelType = function (x) {
            if (!arguments.length) {
                return modelType;
            }
            modelType = x;
            return model;
        };

        model.entities = function (x) {
            if (!arguments.length) {
                return entities;
            }
            entities = x;
            return model;
        };

        model.documents = function (x) {
            if (!arguments.length) {
                return documents;
            }
            documents = x;
            return model;
        };

        model.edges = function (x) {
            if (!arguments.length) {
                return edges;
            }
            edges = x;
            return model;
        };


        // With expressive movements, users are able to inform the system that the weighting vector should be
        // updated to reflect a change in similarity between two (or more) documents.
        // For example, when placing two documents closer together, the system determines the similarity between
        // those two documents, and increases the weight on the corresponding entities.
        // As a result, a new layout is incrementally generated reflecting the new similarity weighting,
        // where those two documents (as well as others sharing similar entities) are closer together.
        model.documentOverlapping = function (docId1, docId2) {
            // Find entities
            console.log("in Document Overlapping");

            var doc1 = documents.find(function (d) {
                return d.id == docId1;
            });

            var doc2 = documents.find(function (d) {
                return d.id == docId2;
            });

            var sharedEntities = doc1.entities.filter(function (e) {
                return doc2.entities.filter(function (ee) {
                        return ee.name == e.name;
                    }).length > 0;
            }).map(function (e) {
                return e.name;
            });

            var keys = Object.keys(entities);
            var decK = sharedEntities.length * K / (keys.length - sharedEntities.length);
            console.log(sharedEntities);
            console.log(decK);
            for (var i in keys) {
                var e = entities[keys[i]];
                if (sharedEntities.indexOf(e.name) >= 0) {
                    console.log(e.name);
                    e.weight += K;
                    console.log(e.weight);
                } else {
                    e.weight -= decK;
                }
            }

            // Update whole mass and strength of spring.
            updateMode();
        };

        // TODO
        model.documentMovement = function (x) {

        };

        // TODO
        model.textHighlighting = function () {

        };

        // TODO
        model.searchTerms = function () {

        };

        // TODO
        model.annotation = function () {

        };


        // TODO
        model.undo = function () {

        };

        return model;
    };


    // If the docs is loaded, init the model: entityWeightVector
    function initModel() {

        console.log('init models');
        // init documents mass
        // update mass
        documents.forEach(function (d) {
            d.mass = d.entities.reduce(function (acc, e) {
                return acc + entities[e.name].weight * e.value;
            }, 0);
        });

        console.log(edges);
        // init edges
        edges.forEach(function (e) {
            console.log(e.strength);

            var d1=documents.find(function (d) {
                return d.id == e.source;
            });

            e.strength = cosineDistance(documents.find(function (d) {
                return d.id == e.source;
            }), documents.find(function (d) {
                return d.id == e.target;
            }), entities);
            console.log(e.strength);
            // edges.strength = cosineDistance(e.source, e.target, entities);
        });

    }


    // When the entities weight updated, update the mass and spring of docs.
    function updateMode() {

        // update mass
        documents.forEach(function (d) {
            d.mass = d.entities.reduce(function (acc, e) {
                return acc + entities[e.name].weight * e.value;
            }, 0);
        });
        // console.log(documents);

        // weighted sum model, to calculate the similarity
        // update edges
        edges.forEach(function (e) {
            console.log(e.strength);
            e.strength = cosineDistance(e.source, e.target, entities);
            console.log(e.strength);
            // edges.strength = cosineDistance(e.source, e.target, entities);
        });
    }

})();

