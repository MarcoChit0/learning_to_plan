(define (problem prob)
 (:domain barman)
 (:objects 
      shaker128 - shaker
      left right - hand
      shot363 - shot
      ingredient233 ingredient444 ingredient405 - ingredient
      cocktail56 - cocktail
      dispenser48 dispenser473 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker128)
  (ontable shot363)
  (dispenses dispenser48 ingredient233)
  (dispenses dispenser473 ingredient444)
  (dispenses dispenser498 ingredient405)
  (clean shaker128)
  (clean shot363)
  (empty shaker128)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker128 l0)
  (shaker-level shaker128 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient233)
  (cocktail-part2 cocktail56 ingredient444)
)
 (:goal
  (and
      (contains shot363 cocktail56)
)))
