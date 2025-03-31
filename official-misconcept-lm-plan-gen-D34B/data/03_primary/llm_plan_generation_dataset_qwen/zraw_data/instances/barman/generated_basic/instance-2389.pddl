(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot95 shot291 - shot
      ingredient165 ingredient272 ingredient179 - ingredient
      cocktail294 - cocktail
      dispenser304 dispenser162 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot95)
  (ontable shot291)
  (dispenses dispenser304 ingredient165)
  (dispenses dispenser162 ingredient272)
  (dispenses dispenser316 ingredient179)
  (clean shaker250)
  (clean shot95)
  (clean shot291)
  (empty shaker250)
  (empty shot95)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail294 ingredient165)
  (cocktail-part2 cocktail294 ingredient272)
)
 (:goal
  (and
      (contains shot95 cocktail294)
)))
