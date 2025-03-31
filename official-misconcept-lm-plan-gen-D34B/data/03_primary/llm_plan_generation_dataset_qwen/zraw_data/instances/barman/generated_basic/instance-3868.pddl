(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot54 shot482 - shot
      ingredient343 ingredient152 ingredient382 - ingredient
      cocktail491 - cocktail
      dispenser164 dispenser231 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot54)
  (ontable shot482)
  (dispenses dispenser164 ingredient343)
  (dispenses dispenser231 ingredient152)
  (dispenses dispenser291 ingredient382)
  (clean shaker468)
  (clean shot54)
  (clean shot482)
  (empty shaker468)
  (empty shot54)
  (empty shot482)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail491 ingredient343)
  (cocktail-part2 cocktail491 ingredient152)
)
 (:goal
  (and
      (contains shot54 cocktail491)
)))
