(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot70 - shot
      ingredient378 ingredient453 ingredient348 ingredient335 - ingredient
      cocktail293 - cocktail
      dispenser414 dispenser313 dispenser267 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot70)
  (dispenses dispenser414 ingredient378)
  (dispenses dispenser313 ingredient453)
  (dispenses dispenser267 ingredient348)
  (dispenses dispenser303 ingredient335)
  (clean shaker49)
  (clean shot70)
  (empty shaker49)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient348)
  (cocktail-part2 cocktail293 ingredient378)
)
 (:goal
  (and
      (contains shot70 cocktail293)
)))
