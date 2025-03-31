(define (problem prob)
 (:domain barman)
 (:objects 
      shaker421 - shaker
      left right - hand
      shot373 - shot
      ingredient168 ingredient302 ingredient282 ingredient313 - ingredient
      cocktail1 - cocktail
      dispenser333 dispenser131 dispenser260 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker421)
  (ontable shot373)
  (dispenses dispenser333 ingredient168)
  (dispenses dispenser131 ingredient302)
  (dispenses dispenser260 ingredient282)
  (dispenses dispenser407 ingredient313)
  (clean shaker421)
  (clean shot373)
  (empty shaker421)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker421 l0)
  (shaker-level shaker421 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient168)
  (cocktail-part2 cocktail1 ingredient282)
)
 (:goal
  (and
      (contains shot373 cocktail1)
)))
