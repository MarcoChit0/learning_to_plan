(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot388 - shot
      ingredient66 ingredient189 ingredient12 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser49 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot388)
  (dispenses dispenser42 ingredient66)
  (dispenses dispenser49 ingredient189)
  (dispenses dispenser62 ingredient12)
  (clean shaker170)
  (clean shot388)
  (empty shaker170)
  (empty shot388)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient189)
  (cocktail-part2 cocktail1 ingredient66)
)
 (:goal
  (and
      (contains shot388 cocktail1)
)))
