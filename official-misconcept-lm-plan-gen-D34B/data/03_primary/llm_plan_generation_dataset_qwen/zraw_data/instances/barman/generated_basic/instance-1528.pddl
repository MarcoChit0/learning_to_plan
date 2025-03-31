(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot493 shot352 - shot
      ingredient58 ingredient353 ingredient31 ingredient290 - ingredient
      cocktail89 - cocktail
      dispenser458 dispenser297 dispenser454 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot493)
  (ontable shot352)
  (dispenses dispenser458 ingredient58)
  (dispenses dispenser297 ingredient353)
  (dispenses dispenser454 ingredient31)
  (dispenses dispenser72 ingredient290)
  (clean shaker136)
  (clean shot493)
  (clean shot352)
  (empty shaker136)
  (empty shot493)
  (empty shot352)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail89 ingredient58)
  (cocktail-part2 cocktail89 ingredient290)
)
 (:goal
  (and
      (contains shot493 cocktail89)
)))
