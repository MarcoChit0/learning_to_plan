(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot108 - shot
      ingredient264 ingredient205 ingredient307 - ingredient
      cocktail169 - cocktail
      dispenser495 dispenser341 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot108)
  (dispenses dispenser495 ingredient264)
  (dispenses dispenser341 ingredient205)
  (dispenses dispenser440 ingredient307)
  (clean shaker25)
  (clean shot108)
  (empty shaker25)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail169 ingredient205)
  (cocktail-part2 cocktail169 ingredient307)
)
 (:goal
  (and
      (contains shot108 cocktail169)
)))
