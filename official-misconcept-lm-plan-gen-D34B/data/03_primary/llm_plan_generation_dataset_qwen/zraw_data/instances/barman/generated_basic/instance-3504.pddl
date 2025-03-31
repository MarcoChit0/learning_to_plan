(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot69 shot421 - shot
      ingredient292 ingredient264 ingredient356 ingredient320 - ingredient
      cocktail318 - cocktail
      dispenser187 dispenser492 dispenser234 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot69)
  (ontable shot421)
  (dispenses dispenser187 ingredient292)
  (dispenses dispenser492 ingredient264)
  (dispenses dispenser234 ingredient356)
  (dispenses dispenser24 ingredient320)
  (clean shaker390)
  (clean shot69)
  (clean shot421)
  (empty shaker390)
  (empty shot69)
  (empty shot421)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail318 ingredient292)
  (cocktail-part2 cocktail318 ingredient264)
)
 (:goal
  (and
      (contains shot69 cocktail318)
)))
