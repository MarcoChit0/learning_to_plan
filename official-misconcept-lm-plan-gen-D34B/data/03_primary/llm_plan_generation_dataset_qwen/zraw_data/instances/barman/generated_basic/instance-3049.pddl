(define (problem prob)
 (:domain barman)
 (:objects 
      shaker35 - shaker
      left right - hand
      shot160 shot284 - shot
      ingredient360 ingredient51 ingredient63 - ingredient
      cocktail222 - cocktail
      dispenser497 dispenser201 dispenser294 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker35)
  (ontable shot160)
  (ontable shot284)
  (dispenses dispenser497 ingredient360)
  (dispenses dispenser201 ingredient51)
  (dispenses dispenser294 ingredient63)
  (clean shaker35)
  (clean shot160)
  (clean shot284)
  (empty shaker35)
  (empty shot160)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker35 l0)
  (shaker-level shaker35 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient63)
  (cocktail-part2 cocktail222 ingredient360)
)
 (:goal
  (and
      (contains shot160 cocktail222)
)))
