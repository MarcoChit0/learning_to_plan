(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot81 - shot
      ingredient477 ingredient331 - ingredient
      cocktail1 - cocktail
      dispenser310 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot81)
  (dispenses dispenser310 ingredient477)
  (dispenses dispenser82 ingredient331)
  (clean shaker13)
  (clean shot81)
  (empty shaker13)
  (empty shot81)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient477)
  (cocktail-part2 cocktail1 ingredient331)
)
 (:goal
  (and
      (contains shot81 cocktail1)
)))
