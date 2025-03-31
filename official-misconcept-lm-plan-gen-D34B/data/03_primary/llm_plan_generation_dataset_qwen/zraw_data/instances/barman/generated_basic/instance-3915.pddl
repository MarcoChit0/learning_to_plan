(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot370 - shot
      ingredient178 ingredient324 ingredient257 ingredient261 - ingredient
      cocktail422 - cocktail
      dispenser253 dispenser128 dispenser130 dispenser459 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot370)
  (dispenses dispenser253 ingredient178)
  (dispenses dispenser128 ingredient324)
  (dispenses dispenser130 ingredient257)
  (dispenses dispenser459 ingredient261)
  (clean shaker63)
  (clean shot370)
  (empty shaker63)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient324)
  (cocktail-part2 cocktail422 ingredient178)
)
 (:goal
  (and
      (contains shot370 cocktail422)
)))
