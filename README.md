# DuckDuckGo Plugin

```python
from ovos_ddg_solver import DuckDuckGoSolver

d = DuckDuckGoSolver()

ans = d.spoken_answer("Quem foi Bartolomeu Dias", lang="pt")
print(ans)
# Bartolomeu Dias, OM, OMP foi um navegador português que ficou célebre por ter sido o primeiro europeu a navegar para além do extremo sul da África, contornando o Cabo da Boa Esperança e chegando ao Oceano Índico a partir do Atlântico, abrindo o caminho marítimo para a Índia. Dele não se conhecem os antepassados, mas mercês e armas a ele outorgadas passaram a seus descendentes. Seu irmão foi Diogo Dias, também experiente navegador. Foi o principal navegador da esquadra de Pedro Álvares Cabral em 1500. As terras do Brasil, até então desconhecidas pelos portugueses, confundiram os navegadores, que pensaram tratar-se de uma ilha, a que deram o nome de "Vera Cruz".

info = d.get_infobox("Stephen Hawking", lang="pt")[0]
from pprint import pprint

pprint(info)
# {'born': 'Quinta-feira, oito de Janeiro, mil novecentos e quarenta e dois',
#  'died': 'Quarta-feira, catorze de Março, dois mil e dezoito',
#  'facebook profile': 'stephenhawking',
#  'imdb id': 'nm0370071',
#  'instance of': {'entity-type': 'item', 'id': 'Q5', 'numeric-id': 5},
#  'official website': 'https://hawking.org.uk',
#  'rotten tomatoes id': 'celebrity/stephen_hawking',
#  'wikidata aliases': ['Stephen Hawking',
#                       'Hawking',
#                       'Stephen William Hawking',
#                       'S. W. Hawking'],
#  'wikidata description': 'físico teórico, cosmólogo e autor inglês (1942–2018)',
#  'wikidata id': 'Q17714',
#  'wikidata label': 'Stephen Hawking',
#  'youtube channel': 'UCPyd4mR0p8zHd8Z0HvHc0fw'}


# chunked answer, "tell me more"
for sentence in d.long_answer("who is Isaac Newton", lang="en"):
    print(sentence["title"])
    print(sentence["summary"])
    print(sentence.get("img"))

    # who is Isaac Newton
    # Sir Isaac Newton was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, author, and inventor.
    # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

    # who is Isaac Newton
    # He was a key figure in the Scientific Revolution and the Enlightenment that followed.
    # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

    # who is Isaac Newton
    # His book Philosophiæ Naturalis Principia Mathematica, first published in 1687, achieved the first great unification in physics and established classical mechanics.
    # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

    # who is Isaac Newton
    # Newton also made seminal contributions to optics, and shares credit with German mathematician Gottfried Wilhelm Leibniz for formulating infinitesimal calculus, though he developed calculus years before Leibniz.
    # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

    # who is Isaac Newton
    # Newton contributed to and refined the scientific method, and his work is considered the most influential in bringing forth modern science.
    # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg

    # who is Isaac Newton
    # In the Principia, Newton formulated the laws of motion and universal gravitation that formed the dominant scientific viewpoint for centuries until it was superseded by the theory of relativity.
    # https://duckduckgo.com/i/401ff0bf4dfa0847.jpg
```