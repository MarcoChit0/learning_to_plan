(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot9 - shot
      ingredient35 ingredient23 - ingredient
      cocktail1 - cocktail
      dispenser88 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot9)
  (dispenses dispenser88 ingredient35)
  (dispenses dispenser387 ingredient23)
  (clean shaker432)
  (clean shot9)
  (empty shaker432)
  (empty shot9)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient35)
  (cocktail-part2 cocktail1 ingredient23)
)
 (:goal
  (and
      (contains shot9 cocktail1)
)))
