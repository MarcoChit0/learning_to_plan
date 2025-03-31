(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot297 shot379 - shot
      ingredient447 ingredient418 ingredient223 - ingredient
      cocktail1 - cocktail
      dispenser283 dispenser296 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot297)
  (ontable shot379)
  (dispenses dispenser283 ingredient447)
  (dispenses dispenser296 ingredient418)
  (dispenses dispenser397 ingredient223)
  (clean shaker380)
  (clean shot297)
  (clean shot379)
  (empty shaker380)
  (empty shot297)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient223)
  (cocktail-part2 cocktail1 ingredient418)
)
 (:goal
  (and
      (contains shot297 cocktail1)
)))
