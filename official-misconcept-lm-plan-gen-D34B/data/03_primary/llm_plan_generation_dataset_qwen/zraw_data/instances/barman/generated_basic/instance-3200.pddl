(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot251 shot12 - shot
      ingredient290 ingredient147 ingredient66 - ingredient
      cocktail343 - cocktail
      dispenser218 dispenser192 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot251)
  (ontable shot12)
  (dispenses dispenser218 ingredient290)
  (dispenses dispenser192 ingredient147)
  (dispenses dispenser116 ingredient66)
  (clean shaker157)
  (clean shot251)
  (clean shot12)
  (empty shaker157)
  (empty shot251)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail343 ingredient147)
  (cocktail-part2 cocktail343 ingredient66)
)
 (:goal
  (and
      (contains shot251 cocktail343)
)))
