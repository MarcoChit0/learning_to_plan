(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot379 - shot
      ingredient176 ingredient399 - ingredient
      cocktail346 - cocktail
      dispenser394 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot379)
  (dispenses dispenser394 ingredient176)
  (dispenses dispenser223 ingredient399)
  (clean shaker451)
  (clean shot379)
  (empty shaker451)
  (empty shot379)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient399)
  (cocktail-part2 cocktail346 ingredient176)
)
 (:goal
  (and
      (contains shot379 cocktail346)
)))
