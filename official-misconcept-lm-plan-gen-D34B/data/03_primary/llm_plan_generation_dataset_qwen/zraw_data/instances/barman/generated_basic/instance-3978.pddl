(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot452 shot479 - shot
      ingredient383 ingredient388 ingredient350 ingredient490 - ingredient
      cocktail156 - cocktail
      dispenser402 dispenser399 dispenser431 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot452)
  (ontable shot479)
  (dispenses dispenser402 ingredient383)
  (dispenses dispenser399 ingredient388)
  (dispenses dispenser431 ingredient350)
  (dispenses dispenser47 ingredient490)
  (clean shaker152)
  (clean shot452)
  (clean shot479)
  (empty shaker152)
  (empty shot452)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient350)
  (cocktail-part2 cocktail156 ingredient383)
)
 (:goal
  (and
      (contains shot452 cocktail156)
)))
