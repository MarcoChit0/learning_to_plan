(define (problem prob)
 (:domain barman)
 (:objects 
      shaker348 - shaker
      left right - hand
      shot455 - shot
      ingredient421 ingredient334 - ingredient
      cocktail1 - cocktail
      dispenser57 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker348)
  (ontable shot455)
  (dispenses dispenser57 ingredient421)
  (dispenses dispenser278 ingredient334)
  (clean shaker348)
  (clean shot455)
  (empty shaker348)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker348 l0)
  (shaker-level shaker348 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient334)
  (cocktail-part2 cocktail1 ingredient421)
)
 (:goal
  (and
      (contains shot455 cocktail1)
)))
