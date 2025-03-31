(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot117 shot109 - shot
      ingredient70 ingredient128 - ingredient
      cocktail320 - cocktail
      dispenser338 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot117)
  (ontable shot109)
  (dispenses dispenser338 ingredient70)
  (dispenses dispenser468 ingredient128)
  (clean shaker85)
  (clean shot117)
  (clean shot109)
  (empty shaker85)
  (empty shot117)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient70)
  (cocktail-part2 cocktail320 ingredient128)
)
 (:goal
  (and
      (contains shot117 cocktail320)
)))
