(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot202 - shot
      ingredient177 ingredient285 ingredient284 ingredient468 - ingredient
      cocktail169 - cocktail
      dispenser480 dispenser249 dispenser456 dispenser491 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot202)
  (dispenses dispenser480 ingredient177)
  (dispenses dispenser249 ingredient285)
  (dispenses dispenser456 ingredient284)
  (dispenses dispenser491 ingredient468)
  (clean shaker477)
  (clean shot202)
  (empty shaker477)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail169 ingredient285)
  (cocktail-part2 cocktail169 ingredient284)
)
 (:goal
  (and
      (contains shot202 cocktail169)
)))
