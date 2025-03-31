(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot42 - shot
      ingredient468 ingredient453 ingredient272 ingredient494 - ingredient
      cocktail167 - cocktail
      dispenser158 dispenser162 dispenser168 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot42)
  (dispenses dispenser158 ingredient468)
  (dispenses dispenser162 ingredient453)
  (dispenses dispenser168 ingredient272)
  (dispenses dispenser26 ingredient494)
  (clean shaker260)
  (clean shot42)
  (empty shaker260)
  (empty shot42)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail167 ingredient272)
  (cocktail-part2 cocktail167 ingredient453)
)
 (:goal
  (and
      (contains shot42 cocktail167)
)))
