(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot62 shot187 - shot
      ingredient462 ingredient121 ingredient60 - ingredient
      cocktail127 - cocktail
      dispenser399 dispenser484 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot62)
  (ontable shot187)
  (dispenses dispenser399 ingredient462)
  (dispenses dispenser484 ingredient121)
  (dispenses dispenser167 ingredient60)
  (clean shaker98)
  (clean shot62)
  (clean shot187)
  (empty shaker98)
  (empty shot62)
  (empty shot187)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail127 ingredient462)
  (cocktail-part2 cocktail127 ingredient60)
)
 (:goal
  (and
      (contains shot62 cocktail127)
)))
