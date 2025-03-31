(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot21 - shot
      ingredient143 ingredient397 - ingredient
      cocktail315 - cocktail
      dispenser214 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot21)
  (dispenses dispenser214 ingredient143)
  (dispenses dispenser497 ingredient397)
  (clean shaker342)
  (clean shot21)
  (empty shaker342)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail315 ingredient397)
  (cocktail-part2 cocktail315 ingredient143)
)
 (:goal
  (and
      (contains shot21 cocktail315)
)))
