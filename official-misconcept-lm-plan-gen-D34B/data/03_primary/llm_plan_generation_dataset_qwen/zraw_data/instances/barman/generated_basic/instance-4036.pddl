(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot488 shot278 - shot
      ingredient277 ingredient41 ingredient282 ingredient335 - ingredient
      cocktail323 - cocktail
      dispenser491 dispenser369 dispenser449 dispenser199 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot488)
  (ontable shot278)
  (dispenses dispenser491 ingredient277)
  (dispenses dispenser369 ingredient41)
  (dispenses dispenser449 ingredient282)
  (dispenses dispenser199 ingredient335)
  (clean shaker13)
  (clean shot488)
  (clean shot278)
  (empty shaker13)
  (empty shot488)
  (empty shot278)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient335)
  (cocktail-part2 cocktail323 ingredient282)
)
 (:goal
  (and
      (contains shot488 cocktail323)
)))
