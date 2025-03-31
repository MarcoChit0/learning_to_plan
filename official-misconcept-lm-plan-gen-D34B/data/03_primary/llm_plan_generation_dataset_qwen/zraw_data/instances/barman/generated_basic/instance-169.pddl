(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot334 shot291 - shot
      ingredient365 ingredient427 ingredient71 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser80 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot334)
  (ontable shot291)
  (dispenses dispenser73 ingredient365)
  (dispenses dispenser80 ingredient427)
  (dispenses dispenser28 ingredient71)
  (clean shaker411)
  (clean shot334)
  (clean shot291)
  (empty shaker411)
  (empty shot334)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient71)
  (cocktail-part2 cocktail1 ingredient427)
)
 (:goal
  (and
      (contains shot334 cocktail1)
)))
