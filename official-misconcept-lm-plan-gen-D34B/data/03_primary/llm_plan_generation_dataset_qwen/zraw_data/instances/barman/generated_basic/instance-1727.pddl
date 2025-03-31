(define (problem prob)
 (:domain barman)
 (:objects 
      shaker175 - shaker
      left right - hand
      shot350 shot65 - shot
      ingredient114 ingredient342 ingredient468 ingredient293 - ingredient
      cocktail284 - cocktail
      dispenser344 dispenser453 dispenser378 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker175)
  (ontable shot350)
  (ontable shot65)
  (dispenses dispenser344 ingredient114)
  (dispenses dispenser453 ingredient342)
  (dispenses dispenser378 ingredient468)
  (dispenses dispenser393 ingredient293)
  (clean shaker175)
  (clean shot350)
  (clean shot65)
  (empty shaker175)
  (empty shot350)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker175 l0)
  (shaker-level shaker175 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail284 ingredient293)
  (cocktail-part2 cocktail284 ingredient468)
)
 (:goal
  (and
      (contains shot350 cocktail284)
)))
