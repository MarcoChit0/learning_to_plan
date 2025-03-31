(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot75 - shot
      ingredient107 ingredient29 ingredient478 ingredient96 - ingredient
      cocktail205 - cocktail
      dispenser56 dispenser57 dispenser22 dispenser392 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot75)
  (dispenses dispenser56 ingredient107)
  (dispenses dispenser57 ingredient29)
  (dispenses dispenser22 ingredient478)
  (dispenses dispenser392 ingredient96)
  (clean shaker325)
  (clean shot75)
  (empty shaker325)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail205 ingredient29)
  (cocktail-part2 cocktail205 ingredient96)
)
 (:goal
  (and
      (contains shot75 cocktail205)
)))
