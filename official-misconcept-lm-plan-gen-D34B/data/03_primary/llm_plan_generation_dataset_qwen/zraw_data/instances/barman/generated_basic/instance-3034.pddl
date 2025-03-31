(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot448 - shot
      ingredient180 ingredient305 ingredient282 - ingredient
      cocktail63 - cocktail
      dispenser68 dispenser128 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot448)
  (dispenses dispenser68 ingredient180)
  (dispenses dispenser128 ingredient305)
  (dispenses dispenser346 ingredient282)
  (clean shaker120)
  (clean shot448)
  (empty shaker120)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient282)
  (cocktail-part2 cocktail63 ingredient305)
)
 (:goal
  (and
      (contains shot448 cocktail63)
)))
