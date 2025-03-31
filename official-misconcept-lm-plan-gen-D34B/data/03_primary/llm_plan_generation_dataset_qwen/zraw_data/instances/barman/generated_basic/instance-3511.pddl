(define (problem prob)
 (:domain barman)
 (:objects 
      shaker455 - shaker
      left right - hand
      shot269 shot109 - shot
      ingredient44 ingredient461 ingredient404 ingredient178 - ingredient
      cocktail101 - cocktail
      dispenser162 dispenser95 dispenser286 dispenser235 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker455)
  (ontable shot269)
  (ontable shot109)
  (dispenses dispenser162 ingredient44)
  (dispenses dispenser95 ingredient461)
  (dispenses dispenser286 ingredient404)
  (dispenses dispenser235 ingredient178)
  (clean shaker455)
  (clean shot269)
  (clean shot109)
  (empty shaker455)
  (empty shot269)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker455 l0)
  (shaker-level shaker455 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient461)
  (cocktail-part2 cocktail101 ingredient178)
)
 (:goal
  (and
      (contains shot269 cocktail101)
)))
