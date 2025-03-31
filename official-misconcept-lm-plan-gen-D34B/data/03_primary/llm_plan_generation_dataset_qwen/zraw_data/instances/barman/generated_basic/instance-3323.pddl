(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot72 shot397 shot452 - shot
      ingredient438 ingredient148 ingredient285 - ingredient
      cocktail308 - cocktail
      dispenser87 dispenser109 dispenser488 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot72)
  (ontable shot397)
  (ontable shot452)
  (dispenses dispenser87 ingredient438)
  (dispenses dispenser109 ingredient148)
  (dispenses dispenser488 ingredient285)
  (clean shaker153)
  (clean shot72)
  (clean shot397)
  (clean shot452)
  (empty shaker153)
  (empty shot72)
  (empty shot397)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient438)
  (cocktail-part2 cocktail308 ingredient285)
)
 (:goal
  (and
      (contains shot72 cocktail308)
      (contains shot397 cocktail308)
)))
