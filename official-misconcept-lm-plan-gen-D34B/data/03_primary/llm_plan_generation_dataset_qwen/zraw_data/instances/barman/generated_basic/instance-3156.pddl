(define (problem prob)
 (:domain barman)
 (:objects 
      shaker182 - shaker
      left right - hand
      shot457 shot119 shot36 - shot
      ingredient383 ingredient243 ingredient255 ingredient264 - ingredient
      cocktail13 - cocktail
      dispenser304 dispenser387 dispenser32 dispenser2 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker182)
  (ontable shot457)
  (ontable shot119)
  (ontable shot36)
  (dispenses dispenser304 ingredient383)
  (dispenses dispenser387 ingredient243)
  (dispenses dispenser32 ingredient255)
  (dispenses dispenser2 ingredient264)
  (clean shaker182)
  (clean shot457)
  (clean shot119)
  (clean shot36)
  (empty shaker182)
  (empty shot457)
  (empty shot119)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker182 l0)
  (shaker-level shaker182 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient255)
  (cocktail-part2 cocktail13 ingredient264)
)
 (:goal
  (and
      (contains shot457 cocktail13)
      (contains shot119 cocktail13)
)))
