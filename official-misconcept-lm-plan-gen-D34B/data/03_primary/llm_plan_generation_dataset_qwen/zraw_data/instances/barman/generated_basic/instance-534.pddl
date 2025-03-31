(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot203 - shot
      ingredient264 ingredient261 ingredient355 - ingredient
      cocktail1 - cocktail
      dispenser426 dispenser78 dispenser238 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot203)
  (dispenses dispenser426 ingredient264)
  (dispenses dispenser78 ingredient261)
  (dispenses dispenser238 ingredient355)
  (clean shaker412)
  (clean shot203)
  (empty shaker412)
  (empty shot203)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient261)
  (cocktail-part2 cocktail1 ingredient264)
)
 (:goal
  (and
      (contains shot203 cocktail1)
)))
