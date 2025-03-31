(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot463 shot399 shot221 - shot
      ingredient77 ingredient191 - ingredient
      cocktail74 - cocktail
      dispenser497 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot463)
  (ontable shot399)
  (ontable shot221)
  (dispenses dispenser497 ingredient77)
  (dispenses dispenser410 ingredient191)
  (clean shaker468)
  (clean shot463)
  (clean shot399)
  (clean shot221)
  (empty shaker468)
  (empty shot463)
  (empty shot399)
  (empty shot221)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient191)
  (cocktail-part2 cocktail74 ingredient77)
)
 (:goal
  (and
      (contains shot463 cocktail74)
      (contains shot399 cocktail74)
)))
