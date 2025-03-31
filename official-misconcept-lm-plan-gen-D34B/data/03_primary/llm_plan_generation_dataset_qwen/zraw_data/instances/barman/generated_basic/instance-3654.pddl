(define (problem prob)
 (:domain barman)
 (:objects 
      shaker408 - shaker
      left right - hand
      shot1 shot5 - shot
      ingredient353 ingredient10 ingredient361 ingredient330 - ingredient
      cocktail395 - cocktail
      dispenser334 dispenser440 dispenser104 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker408)
  (ontable shot1)
  (ontable shot5)
  (dispenses dispenser334 ingredient353)
  (dispenses dispenser440 ingredient10)
  (dispenses dispenser104 ingredient361)
  (dispenses dispenser162 ingredient330)
  (clean shaker408)
  (clean shot1)
  (clean shot5)
  (empty shaker408)
  (empty shot1)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker408 l0)
  (shaker-level shaker408 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail395 ingredient353)
  (cocktail-part2 cocktail395 ingredient10)
)
 (:goal
  (and
      (contains shot1 cocktail395)
)))
