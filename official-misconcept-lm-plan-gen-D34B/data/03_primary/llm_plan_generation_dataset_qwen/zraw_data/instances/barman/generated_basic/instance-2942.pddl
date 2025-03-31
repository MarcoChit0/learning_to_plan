(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot37 - shot
      ingredient103 ingredient343 ingredient344 ingredient150 - ingredient
      cocktail352 - cocktail
      dispenser340 dispenser257 dispenser78 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot37)
  (dispenses dispenser340 ingredient103)
  (dispenses dispenser257 ingredient343)
  (dispenses dispenser78 ingredient344)
  (dispenses dispenser247 ingredient150)
  (clean shaker110)
  (clean shot37)
  (empty shaker110)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient150)
  (cocktail-part2 cocktail352 ingredient103)
)
 (:goal
  (and
      (contains shot37 cocktail352)
)))
