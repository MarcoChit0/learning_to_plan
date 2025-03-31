(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot284 shot375 - shot
      ingredient167 ingredient475 - ingredient
      cocktail224 - cocktail
      dispenser204 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot284)
  (ontable shot375)
  (dispenses dispenser204 ingredient167)
  (dispenses dispenser73 ingredient475)
  (clean shaker72)
  (clean shot284)
  (clean shot375)
  (empty shaker72)
  (empty shot284)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail224 ingredient475)
  (cocktail-part2 cocktail224 ingredient167)
)
 (:goal
  (and
      (contains shot284 cocktail224)
)))
