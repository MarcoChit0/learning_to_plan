(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot133 shot350 - shot
      ingredient264 ingredient374 ingredient53 ingredient279 - ingredient
      cocktail164 - cocktail
      dispenser284 dispenser13 dispenser377 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot133)
  (ontable shot350)
  (dispenses dispenser284 ingredient264)
  (dispenses dispenser13 ingredient374)
  (dispenses dispenser377 ingredient53)
  (dispenses dispenser32 ingredient279)
  (clean shaker228)
  (clean shot133)
  (clean shot350)
  (empty shaker228)
  (empty shot133)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient279)
  (cocktail-part2 cocktail164 ingredient53)
)
 (:goal
  (and
      (contains shot133 cocktail164)
)))
