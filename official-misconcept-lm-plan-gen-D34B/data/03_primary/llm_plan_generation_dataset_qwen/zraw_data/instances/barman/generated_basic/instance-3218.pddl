(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot103 - shot
      ingredient53 ingredient3 - ingredient
      cocktail131 - cocktail
      dispenser188 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot103)
  (dispenses dispenser188 ingredient53)
  (dispenses dispenser71 ingredient3)
  (clean shaker99)
  (clean shot103)
  (empty shaker99)
  (empty shot103)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient53)
  (cocktail-part2 cocktail131 ingredient3)
)
 (:goal
  (and
      (contains shot103 cocktail131)
)))
