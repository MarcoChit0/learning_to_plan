(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot495 shot452 shot267 - shot
      ingredient290 ingredient297 ingredient203 ingredient365 - ingredient
      cocktail244 - cocktail
      dispenser373 dispenser297 dispenser188 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot495)
  (ontable shot452)
  (ontable shot267)
  (dispenses dispenser373 ingredient290)
  (dispenses dispenser297 ingredient297)
  (dispenses dispenser188 ingredient203)
  (dispenses dispenser390 ingredient365)
  (clean shaker342)
  (clean shot495)
  (clean shot452)
  (clean shot267)
  (empty shaker342)
  (empty shot495)
  (empty shot452)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail244 ingredient297)
  (cocktail-part2 cocktail244 ingredient290)
)
 (:goal
  (and
      (contains shot495 cocktail244)
      (contains shot452 cocktail244)
)))
