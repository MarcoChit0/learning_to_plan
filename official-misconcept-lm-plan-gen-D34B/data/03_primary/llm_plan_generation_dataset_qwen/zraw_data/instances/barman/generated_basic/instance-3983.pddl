(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot454 shot6 - shot
      ingredient237 ingredient146 ingredient421 ingredient335 - ingredient
      cocktail402 - cocktail
      dispenser5 dispenser99 dispenser397 dispenser76 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot454)
  (ontable shot6)
  (dispenses dispenser5 ingredient237)
  (dispenses dispenser99 ingredient146)
  (dispenses dispenser397 ingredient421)
  (dispenses dispenser76 ingredient335)
  (clean shaker465)
  (clean shot454)
  (clean shot6)
  (empty shaker465)
  (empty shot454)
  (empty shot6)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient237)
  (cocktail-part2 cocktail402 ingredient421)
)
 (:goal
  (and
      (contains shot454 cocktail402)
)))
