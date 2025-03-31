(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot450 - shot
      ingredient257 ingredient195 - ingredient
      cocktail39 - cocktail
      dispenser277 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot450)
  (dispenses dispenser277 ingredient257)
  (dispenses dispenser274 ingredient195)
  (clean shaker241)
  (clean shot450)
  (empty shaker241)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient195)
  (cocktail-part2 cocktail39 ingredient257)
)
 (:goal
  (and
      (contains shot450 cocktail39)
)))
