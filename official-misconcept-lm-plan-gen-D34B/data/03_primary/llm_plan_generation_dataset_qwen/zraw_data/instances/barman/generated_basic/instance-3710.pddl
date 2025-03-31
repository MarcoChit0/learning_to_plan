(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot308 shot106 shot484 shot418 - shot
      ingredient197 ingredient450 - ingredient
      cocktail357 - cocktail
      dispenser103 dispenser123 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot308)
  (ontable shot106)
  (ontable shot484)
  (ontable shot418)
  (dispenses dispenser103 ingredient197)
  (dispenses dispenser123 ingredient450)
  (clean shaker358)
  (clean shot308)
  (clean shot106)
  (clean shot484)
  (clean shot418)
  (empty shaker358)
  (empty shot308)
  (empty shot106)
  (empty shot484)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient197)
  (cocktail-part2 cocktail357 ingredient450)
)
 (:goal
  (and
      (contains shot308 cocktail357)
      (contains shot106 ingredient450)
      (contains shot484 cocktail357)
)))
