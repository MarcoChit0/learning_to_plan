(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot7 shot253 - shot
      ingredient252 ingredient481 ingredient373 ingredient499 - ingredient
      cocktail1 - cocktail
      dispenser122 dispenser255 dispenser428 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot7)
  (ontable shot253)
  (dispenses dispenser122 ingredient252)
  (dispenses dispenser255 ingredient481)
  (dispenses dispenser428 ingredient373)
  (dispenses dispenser21 ingredient499)
  (clean shaker262)
  (clean shot7)
  (clean shot253)
  (empty shaker262)
  (empty shot7)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient373)
  (cocktail-part2 cocktail1 ingredient481)
)
 (:goal
  (and
      (contains shot7 cocktail1)
)))
