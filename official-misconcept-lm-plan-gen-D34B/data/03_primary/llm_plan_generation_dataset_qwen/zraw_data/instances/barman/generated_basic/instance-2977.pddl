(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot98 - shot
      ingredient272 ingredient140 - ingredient
      cocktail415 - cocktail
      dispenser163 dispenser334 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot98)
  (dispenses dispenser163 ingredient272)
  (dispenses dispenser334 ingredient140)
  (clean shaker468)
  (clean shot98)
  (empty shaker468)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail415 ingredient272)
  (cocktail-part2 cocktail415 ingredient140)
)
 (:goal
  (and
      (contains shot98 cocktail415)
)))
