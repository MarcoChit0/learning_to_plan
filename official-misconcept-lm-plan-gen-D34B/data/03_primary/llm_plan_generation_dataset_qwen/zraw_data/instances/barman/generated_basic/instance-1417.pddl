(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot494 shot317 - shot
      ingredient308 ingredient399 - ingredient
      cocktail468 - cocktail
      dispenser82 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot494)
  (ontable shot317)
  (dispenses dispenser82 ingredient308)
  (dispenses dispenser156 ingredient399)
  (clean shaker432)
  (clean shot494)
  (clean shot317)
  (empty shaker432)
  (empty shot494)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient399)
  (cocktail-part2 cocktail468 ingredient308)
)
 (:goal
  (and
      (contains shot494 cocktail468)
)))
