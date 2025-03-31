(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot61 - shot
      ingredient254 ingredient46 ingredient188 ingredient431 - ingredient
      cocktail108 - cocktail
      dispenser15 dispenser205 dispenser335 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot61)
  (dispenses dispenser15 ingredient254)
  (dispenses dispenser205 ingredient46)
  (dispenses dispenser335 ingredient188)
  (dispenses dispenser128 ingredient431)
  (clean shaker458)
  (clean shot61)
  (empty shaker458)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient46)
  (cocktail-part2 cocktail108 ingredient254)
)
 (:goal
  (and
      (contains shot61 cocktail108)
)))
