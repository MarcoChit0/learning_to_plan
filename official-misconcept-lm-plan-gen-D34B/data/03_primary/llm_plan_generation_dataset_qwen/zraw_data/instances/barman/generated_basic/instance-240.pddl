(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot387 - shot
      ingredient383 ingredient101 - ingredient
      cocktail1 - cocktail
      dispenser389 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot387)
  (dispenses dispenser389 ingredient383)
  (dispenses dispenser200 ingredient101)
  (clean shaker496)
  (clean shot387)
  (empty shaker496)
  (empty shot387)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient383)
  (cocktail-part2 cocktail1 ingredient101)
)
 (:goal
  (and
      (contains shot387 cocktail1)
)))
