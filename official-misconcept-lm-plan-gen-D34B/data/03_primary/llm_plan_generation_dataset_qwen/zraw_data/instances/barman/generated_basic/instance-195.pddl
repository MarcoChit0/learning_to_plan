(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot128 - shot
      ingredient182 ingredient419 ingredient402 - ingredient
      cocktail1 - cocktail
      dispenser429 dispenser369 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot128)
  (dispenses dispenser429 ingredient182)
  (dispenses dispenser369 ingredient419)
  (dispenses dispenser233 ingredient402)
  (clean shaker365)
  (clean shot128)
  (empty shaker365)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient419)
  (cocktail-part2 cocktail1 ingredient402)
)
 (:goal
  (and
      (contains shot128 cocktail1)
)))
