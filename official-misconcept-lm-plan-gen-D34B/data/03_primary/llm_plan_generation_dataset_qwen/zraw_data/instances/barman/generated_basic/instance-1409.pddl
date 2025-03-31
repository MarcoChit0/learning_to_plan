(define (problem prob)
 (:domain barman)
 (:objects 
      shaker414 - shaker
      left right - hand
      shot453 - shot
      ingredient121 ingredient76 ingredient232 ingredient159 - ingredient
      cocktail125 - cocktail
      dispenser33 dispenser460 dispenser432 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker414)
  (ontable shot453)
  (dispenses dispenser33 ingredient121)
  (dispenses dispenser460 ingredient76)
  (dispenses dispenser432 ingredient232)
  (dispenses dispenser7 ingredient159)
  (clean shaker414)
  (clean shot453)
  (empty shaker414)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker414 l0)
  (shaker-level shaker414 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail125 ingredient76)
  (cocktail-part2 cocktail125 ingredient159)
)
 (:goal
  (and
      (contains shot453 cocktail125)
)))
