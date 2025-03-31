(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot363 - shot
      ingredient60 ingredient145 ingredient318 - ingredient
      cocktail352 - cocktail
      dispenser251 dispenser137 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot363)
  (dispenses dispenser251 ingredient60)
  (dispenses dispenser137 ingredient145)
  (dispenses dispenser46 ingredient318)
  (clean shaker103)
  (clean shot363)
  (empty shaker103)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient318)
  (cocktail-part2 cocktail352 ingredient145)
)
 (:goal
  (and
      (contains shot363 cocktail352)
)))
