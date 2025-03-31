(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot218 - shot
      ingredient245 ingredient477 ingredient374 - ingredient
      cocktail49 - cocktail
      dispenser124 dispenser318 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot218)
  (dispenses dispenser124 ingredient245)
  (dispenses dispenser318 ingredient477)
  (dispenses dispenser57 ingredient374)
  (clean shaker467)
  (clean shot218)
  (empty shaker467)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient374)
  (cocktail-part2 cocktail49 ingredient477)
)
 (:goal
  (and
      (contains shot218 cocktail49)
)))
