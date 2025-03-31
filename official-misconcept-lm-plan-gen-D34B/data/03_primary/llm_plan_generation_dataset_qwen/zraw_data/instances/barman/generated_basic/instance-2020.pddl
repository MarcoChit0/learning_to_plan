(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot477 shot483 - shot
      ingredient229 ingredient74 - ingredient
      cocktail200 - cocktail
      dispenser146 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot477)
  (ontable shot483)
  (dispenses dispenser146 ingredient229)
  (dispenses dispenser40 ingredient74)
  (clean shaker391)
  (clean shot477)
  (clean shot483)
  (empty shaker391)
  (empty shot477)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient229)
  (cocktail-part2 cocktail200 ingredient74)
)
 (:goal
  (and
      (contains shot477 cocktail200)
)))
