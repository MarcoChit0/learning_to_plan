(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot318 shot251 - shot
      ingredient431 ingredient12 ingredient118 - ingredient
      cocktail1 - cocktail
      dispenser4 dispenser386 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot318)
  (ontable shot251)
  (dispenses dispenser4 ingredient431)
  (dispenses dispenser386 ingredient12)
  (dispenses dispenser162 ingredient118)
  (clean shaker341)
  (clean shot318)
  (clean shot251)
  (empty shaker341)
  (empty shot318)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient431)
  (cocktail-part2 cocktail1 ingredient118)
)
 (:goal
  (and
      (contains shot318 cocktail1)
)))
