(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot41 - shot
      ingredient244 ingredient282 ingredient411 ingredient264 - ingredient
      cocktail356 - cocktail
      dispenser150 dispenser116 dispenser194 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot41)
  (dispenses dispenser150 ingredient244)
  (dispenses dispenser116 ingredient282)
  (dispenses dispenser194 ingredient411)
  (dispenses dispenser440 ingredient264)
  (clean shaker286)
  (clean shot41)
  (empty shaker286)
  (empty shot41)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient411)
  (cocktail-part2 cocktail356 ingredient264)
)
 (:goal
  (and
      (contains shot41 cocktail356)
)))
