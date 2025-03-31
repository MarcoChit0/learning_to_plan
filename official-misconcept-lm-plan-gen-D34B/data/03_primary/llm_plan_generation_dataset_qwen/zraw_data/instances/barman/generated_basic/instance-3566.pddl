(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot358 shot449 - shot
      ingredient264 ingredient496 ingredient296 ingredient323 - ingredient
      cocktail297 - cocktail
      dispenser28 dispenser29 dispenser448 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot358)
  (ontable shot449)
  (dispenses dispenser28 ingredient264)
  (dispenses dispenser29 ingredient496)
  (dispenses dispenser448 ingredient296)
  (dispenses dispenser333 ingredient323)
  (clean shaker49)
  (clean shot358)
  (clean shot449)
  (empty shaker49)
  (empty shot358)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail297 ingredient496)
  (cocktail-part2 cocktail297 ingredient264)
)
 (:goal
  (and
      (contains shot358 cocktail297)
)))
