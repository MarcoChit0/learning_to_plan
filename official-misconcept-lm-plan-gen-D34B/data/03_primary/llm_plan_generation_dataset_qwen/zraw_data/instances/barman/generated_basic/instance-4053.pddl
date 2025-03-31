(define (problem prob)
 (:domain barman)
 (:objects 
      shaker421 - shaker
      left right - hand
      shot264 shot424 - shot
      ingredient372 ingredient4 ingredient362 ingredient475 - ingredient
      cocktail371 - cocktail
      dispenser239 dispenser448 dispenser173 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker421)
  (ontable shot264)
  (ontable shot424)
  (dispenses dispenser239 ingredient372)
  (dispenses dispenser448 ingredient4)
  (dispenses dispenser173 ingredient362)
  (dispenses dispenser200 ingredient475)
  (clean shaker421)
  (clean shot264)
  (clean shot424)
  (empty shaker421)
  (empty shot264)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker421 l0)
  (shaker-level shaker421 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient4)
  (cocktail-part2 cocktail371 ingredient475)
)
 (:goal
  (and
      (contains shot264 cocktail371)
)))
