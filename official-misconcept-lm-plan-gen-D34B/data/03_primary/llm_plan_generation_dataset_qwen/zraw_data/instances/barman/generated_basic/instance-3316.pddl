(define (problem prob)
 (:domain barman)
 (:objects 
      shaker256 - shaker
      left right - hand
      shot79 shot59 shot384 - shot
      ingredient494 ingredient67 ingredient215 - ingredient
      cocktail250 - cocktail
      dispenser396 dispenser122 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker256)
  (ontable shot79)
  (ontable shot59)
  (ontable shot384)
  (dispenses dispenser396 ingredient494)
  (dispenses dispenser122 ingredient67)
  (dispenses dispenser237 ingredient215)
  (clean shaker256)
  (clean shot79)
  (clean shot59)
  (clean shot384)
  (empty shaker256)
  (empty shot79)
  (empty shot59)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker256 l0)
  (shaker-level shaker256 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient494)
  (cocktail-part2 cocktail250 ingredient215)
)
 (:goal
  (and
      (contains shot79 cocktail250)
      (contains shot59 ingredient215)
)))
