(define (problem prob)
 (:domain barman)
 (:objects 
      shaker203 - shaker
      left right - hand
      shot31 - shot
      ingredient39 ingredient211 ingredient194 ingredient302 - ingredient
      cocktail16 - cocktail
      dispenser40 dispenser382 dispenser168 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker203)
  (ontable shot31)
  (dispenses dispenser40 ingredient39)
  (dispenses dispenser382 ingredient211)
  (dispenses dispenser168 ingredient194)
  (dispenses dispenser266 ingredient302)
  (clean shaker203)
  (clean shot31)
  (empty shaker203)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker203 l0)
  (shaker-level shaker203 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail16 ingredient302)
  (cocktail-part2 cocktail16 ingredient211)
)
 (:goal
  (and
      (contains shot31 cocktail16)
)))
