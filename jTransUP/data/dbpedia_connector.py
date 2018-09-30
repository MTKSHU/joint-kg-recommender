
from SPARQLWrapper import SPARQLWrapper, N3
from rdflib import Graph

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
    SELECT * WHERE {
{<http://dbpedia.org/resource/One_Flew_Over_the_Cuckoo's_Nest_(film)> ?p ?o}
UNION
{?s ?p <http://dbpedia.org/resource/One_Flew_Over_the_Cuckoo's_Nest_(film)>}
}
""")

sparql.setReturnFormat(N3)
results = sparql.query().convert()
g = Graph()
g.parse(data=results, format="n3")
print(g.serialize(format='n3'))
