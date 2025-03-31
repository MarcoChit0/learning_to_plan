(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot50 shot132 - shot
      ingredient380 ingredient338 - ingredient
      cocktail165 - cocktail
      dispenser456 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot50)
  (ontable shot132)
  (dispenses dispenser456 ingredient380)
  (dispenses dispenser422 ingredient338)
  (clean shaker270)
  (clean shot50)
  (clean shot132)
  (empty shaker270)
  (empty shot50)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient338)
  (cocktail-part2 cocktail165 ingredient380)
)
 (:goal
  (and
      (contains shot50 cocktail165)
)))
