(define (problem prob)
 (:domain barman)
 (:objects 
      shaker121 - shaker
      left right - hand
      shot90 - shot
      ingredient337 ingredient334 - ingredient
      cocktail241 - cocktail
      dispenser341 dispenser172 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker121)
  (ontable shot90)
  (dispenses dispenser341 ingredient337)
  (dispenses dispenser172 ingredient334)
  (clean shaker121)
  (clean shot90)
  (empty shaker121)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker121 l0)
  (shaker-level shaker121 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient334)
  (cocktail-part2 cocktail241 ingredient337)
)
 (:goal
  (and
      (contains shot90 cocktail241)
)))
