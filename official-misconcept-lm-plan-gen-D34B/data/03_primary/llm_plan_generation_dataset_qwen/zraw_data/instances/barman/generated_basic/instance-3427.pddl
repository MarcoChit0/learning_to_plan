(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot360 shot131 - shot
      ingredient479 ingredient231 ingredient388 - ingredient
      cocktail262 - cocktail
      dispenser112 dispenser39 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot360)
  (ontable shot131)
  (dispenses dispenser112 ingredient479)
  (dispenses dispenser39 ingredient231)
  (dispenses dispenser33 ingredient388)
  (clean shaker1)
  (clean shot360)
  (clean shot131)
  (empty shaker1)
  (empty shot360)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient479)
  (cocktail-part2 cocktail262 ingredient231)
)
 (:goal
  (and
      (contains shot360 cocktail262)
)))
