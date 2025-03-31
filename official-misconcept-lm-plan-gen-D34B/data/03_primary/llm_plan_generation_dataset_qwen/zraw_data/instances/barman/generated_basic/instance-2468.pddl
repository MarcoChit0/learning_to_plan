(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot217 - shot
      ingredient27 ingredient405 ingredient380 - ingredient
      cocktail290 - cocktail
      dispenser244 dispenser327 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot217)
  (dispenses dispenser244 ingredient27)
  (dispenses dispenser327 ingredient405)
  (dispenses dispenser490 ingredient380)
  (clean shaker408)
  (clean shot217)
  (empty shaker408)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient27)
  (cocktail-part2 cocktail290 ingredient405)
)
 (:goal
  (and
      (contains shot217 cocktail290)
)))
