(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot397 shot399 - shot
      ingredient480 ingredient316 ingredient73 - ingredient
      cocktail264 - cocktail
      dispenser499 dispenser244 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot397)
  (ontable shot399)
  (dispenses dispenser499 ingredient480)
  (dispenses dispenser244 ingredient316)
  (dispenses dispenser64 ingredient73)
  (clean shaker451)
  (clean shot397)
  (clean shot399)
  (empty shaker451)
  (empty shot397)
  (empty shot399)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail264 ingredient73)
  (cocktail-part2 cocktail264 ingredient316)
)
 (:goal
  (and
      (contains shot397 cocktail264)
)))
