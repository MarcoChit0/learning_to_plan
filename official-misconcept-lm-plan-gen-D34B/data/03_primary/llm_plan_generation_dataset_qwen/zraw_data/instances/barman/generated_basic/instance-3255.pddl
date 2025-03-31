(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot95 shot71 shot120 - shot
      ingredient339 ingredient227 ingredient272 ingredient6 - ingredient
      cocktail57 - cocktail
      dispenser22 dispenser348 dispenser442 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot95)
  (ontable shot71)
  (ontable shot120)
  (dispenses dispenser22 ingredient339)
  (dispenses dispenser348 ingredient227)
  (dispenses dispenser442 ingredient272)
  (dispenses dispenser103 ingredient6)
  (clean shaker271)
  (clean shot95)
  (clean shot71)
  (clean shot120)
  (empty shaker271)
  (empty shot95)
  (empty shot71)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail57 ingredient6)
  (cocktail-part2 cocktail57 ingredient272)
)
 (:goal
  (and
      (contains shot95 cocktail57)
      (contains shot71 cocktail57)
)))
