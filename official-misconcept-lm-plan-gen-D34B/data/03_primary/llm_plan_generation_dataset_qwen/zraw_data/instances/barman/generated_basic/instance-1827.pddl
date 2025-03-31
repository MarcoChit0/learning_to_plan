(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot399 shot245 shot118 - shot
      ingredient487 ingredient358 ingredient441 ingredient232 - ingredient
      cocktail82 - cocktail
      dispenser275 dispenser301 dispenser428 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot399)
  (ontable shot245)
  (ontable shot118)
  (dispenses dispenser275 ingredient487)
  (dispenses dispenser301 ingredient358)
  (dispenses dispenser428 ingredient441)
  (dispenses dispenser195 ingredient232)
  (clean shaker385)
  (clean shot399)
  (clean shot245)
  (clean shot118)
  (empty shaker385)
  (empty shot399)
  (empty shot245)
  (empty shot118)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient358)
  (cocktail-part2 cocktail82 ingredient441)
)
 (:goal
  (and
      (contains shot399 cocktail82)
      (contains shot245 ingredient487)
)))
