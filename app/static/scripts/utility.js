/**
 * Created by Yali on 2/12/17.
 */


// Check if two rectangle overlap each other
// The format of rectangle is {x: 0, y: 0, width: 10, height: 10}
function rectOverlap(A, B) {

    function valueInRange(value, min, max) {
        return (value <= max) && (value >= min);
    }

    var xOverlap = valueInRange(A.x, B.x, B.x + B.width) ||
        valueInRange(B.x, A.x, A.x + A.width);

    var yOverlap = valueInRange(A.y, B.y, B.y + B.height) ||
        valueInRange(B.y, A.y, A.y + A.height);

    return xOverlap && yOverlap;
}


// Moving an SVG selection to the front/back
// Thanks to d3-extended (github.com/wbkd/d3-extended)
d3.selection.prototype.moveToFront = function () {
    return this.each(function () {
        this.parentNode.appendChild(this);
    });
};

d3.selection.prototype.moveToBack = function () {
    return this.each(function () {
        var firstChild = this.parentNode.firstChild;
        if (firstChild) {
            this.parentNode.insertBefore(this, firstChild);
        }
    });
};



// Calculate the distance between two documents based on entity weights

// The normalization must also be weighted. For vectors u and v,
// with weight vector w, the weighted cosine is
// (sum w[i]*u[i]*v[i]) / sqrt[(sum w[i]*u[i]^2)*(sum w[i]*v[i]^2)].
function cosineDistance(doc1, doc2, entities) {

    var distance = 0.0;
    var len1 = 0.0;
    var len2 = 0.0;

    doc1.entities.forEach(function (e1) {

        len1 += e1.value * e1.value * entities[e1.name].weight;
        doc2.entities.forEach(function (e2) {
            len2 += e2.value * e2.value * entities[e2.name].weight;
            if (e1.name == e2.name) {
                distance += e1.value * e2.value * entities[e1.name].weight;
            }
        });

    });

    len1 = Math.sqrt(len1);
    len2 = Math.sqrt(len2);

    return distance/(len1 * len2);
}


