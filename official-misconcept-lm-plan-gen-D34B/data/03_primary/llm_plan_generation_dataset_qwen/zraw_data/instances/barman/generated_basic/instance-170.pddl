(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot334 shot84 - shot
      ingredient454 ingredient0 ingredient136 - ingredient
      cocktail1 - cocktail
      dispenser110 dispenser413 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot334)
  (ontable shot84)
  (dispenses dispenser110 ingredient454)
  (dispenses dispenser413 ingredient0)
  (dispenses dispenser149 ingredient136)
  (clean shaker99)
  (clean shot334)
  (clean shot84)
  (empty shaker99)
  (empty shot334)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient454)
  (cocktail-part2 cocktail1 ingredient0)
)
 (:goal
  (and
      (contains shot334 cocktail1)
)))
