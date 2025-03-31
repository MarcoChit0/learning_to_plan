(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot437 shot450 - shot
      ingredient382 ingredient117 ingredient364 - ingredient
      cocktail29 - cocktail
      dispenser350 dispenser152 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot437)
  (ontable shot450)
  (dispenses dispenser350 ingredient382)
  (dispenses dispenser152 ingredient117)
  (dispenses dispenser381 ingredient364)
  (clean shaker91)
  (clean shot437)
  (clean shot450)
  (empty shaker91)
  (empty shot437)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient364)
  (cocktail-part2 cocktail29 ingredient117)
)
 (:goal
  (and
      (contains shot437 cocktail29)
)))
