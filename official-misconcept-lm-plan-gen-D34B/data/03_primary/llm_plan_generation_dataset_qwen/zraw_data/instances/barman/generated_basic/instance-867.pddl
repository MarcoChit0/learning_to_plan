(define (problem prob)
 (:domain barman)
 (:objects 
      shaker131 - shaker
      left right - hand
      shot65 shot48 shot397 - shot
      ingredient63 ingredient227 ingredient66 ingredient167 - ingredient
      cocktail1 - cocktail
      dispenser97 dispenser412 dispenser214 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker131)
  (ontable shot65)
  (ontable shot48)
  (ontable shot397)
  (dispenses dispenser97 ingredient63)
  (dispenses dispenser412 ingredient227)
  (dispenses dispenser214 ingredient66)
  (dispenses dispenser278 ingredient167)
  (clean shaker131)
  (clean shot65)
  (clean shot48)
  (clean shot397)
  (empty shaker131)
  (empty shot65)
  (empty shot48)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker131 l0)
  (shaker-level shaker131 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient167)
  (cocktail-part2 cocktail1 ingredient227)
)
 (:goal
  (and
      (contains shot65 cocktail1)
      (contains shot48 cocktail1)
)))
