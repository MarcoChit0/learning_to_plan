(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot118 shot214 - shot
      ingredient360 ingredient203 - ingredient
      cocktail499 - cocktail
      dispenser253 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot118)
  (ontable shot214)
  (dispenses dispenser253 ingredient360)
  (dispenses dispenser212 ingredient203)
  (clean shaker408)
  (clean shot118)
  (clean shot214)
  (empty shaker408)
  (empty shot118)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail499 ingredient203)
  (cocktail-part2 cocktail499 ingredient360)
)
 (:goal
  (and
      (contains shot118 cocktail499)
)))
