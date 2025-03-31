(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot82 - shot
      ingredient395 ingredient469 - ingredient
      cocktail319 - cocktail
      dispenser122 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot82)
  (dispenses dispenser122 ingredient395)
  (dispenses dispenser255 ingredient469)
  (clean shaker264)
  (clean shot82)
  (empty shaker264)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient469)
  (cocktail-part2 cocktail319 ingredient395)
)
 (:goal
  (and
      (contains shot82 cocktail319)
)))
