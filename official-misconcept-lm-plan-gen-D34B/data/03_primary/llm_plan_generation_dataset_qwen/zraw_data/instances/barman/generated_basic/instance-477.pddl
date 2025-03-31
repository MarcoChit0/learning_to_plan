(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot232 shot408 - shot
      ingredient98 ingredient45 ingredient96 ingredient214 - ingredient
      cocktail1 - cocktail
      dispenser185 dispenser464 dispenser23 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot232)
  (ontable shot408)
  (dispenses dispenser185 ingredient98)
  (dispenses dispenser464 ingredient45)
  (dispenses dispenser23 ingredient96)
  (dispenses dispenser480 ingredient214)
  (clean shaker305)
  (clean shot232)
  (clean shot408)
  (empty shaker305)
  (empty shot232)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient214)
  (cocktail-part2 cocktail1 ingredient98)
)
 (:goal
  (and
      (contains shot232 cocktail1)
)))
