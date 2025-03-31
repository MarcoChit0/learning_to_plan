(define (problem prob)
 (:domain barman)
 (:objects 
      shaker196 - shaker
      left right - hand
      shot162 shot402 - shot
      ingredient293 ingredient292 ingredient29 ingredient227 - ingredient
      cocktail81 - cocktail
      dispenser490 dispenser47 dispenser401 dispenser335 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker196)
  (ontable shot162)
  (ontable shot402)
  (dispenses dispenser490 ingredient293)
  (dispenses dispenser47 ingredient292)
  (dispenses dispenser401 ingredient29)
  (dispenses dispenser335 ingredient227)
  (clean shaker196)
  (clean shot162)
  (clean shot402)
  (empty shaker196)
  (empty shot162)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker196 l0)
  (shaker-level shaker196 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient293)
  (cocktail-part2 cocktail81 ingredient227)
)
 (:goal
  (and
      (contains shot162 cocktail81)
)))
