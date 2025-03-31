(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot375 - shot
      ingredient66 ingredient333 ingredient372 ingredient244 - ingredient
      cocktail1 - cocktail
      dispenser470 dispenser103 dispenser91 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot375)
  (dispenses dispenser470 ingredient66)
  (dispenses dispenser103 ingredient333)
  (dispenses dispenser91 ingredient372)
  (dispenses dispenser14 ingredient244)
  (clean shaker220)
  (clean shot375)
  (empty shaker220)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient372)
  (cocktail-part2 cocktail1 ingredient333)
)
 (:goal
  (and
      (contains shot375 cocktail1)
)))
