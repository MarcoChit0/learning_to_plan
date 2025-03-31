(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot379 - shot
      ingredient347 ingredient77 ingredient108 - ingredient
      cocktail1 - cocktail
      dispenser235 dispenser353 dispenser185 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot379)
  (dispenses dispenser235 ingredient347)
  (dispenses dispenser353 ingredient77)
  (dispenses dispenser185 ingredient108)
  (clean shaker49)
  (clean shot379)
  (empty shaker49)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient347)
  (cocktail-part2 cocktail1 ingredient108)
)
 (:goal
  (and
      (contains shot379 cocktail1)
)))
