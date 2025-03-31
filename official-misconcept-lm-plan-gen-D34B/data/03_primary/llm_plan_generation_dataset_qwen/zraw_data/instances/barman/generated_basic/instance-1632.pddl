(define (problem prob)
 (:domain barman)
 (:objects 
      shaker352 - shaker
      left right - hand
      shot418 shot148 shot264 - shot
      ingredient166 ingredient485 - ingredient
      cocktail192 - cocktail
      dispenser169 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker352)
  (ontable shot418)
  (ontable shot148)
  (ontable shot264)
  (dispenses dispenser169 ingredient166)
  (dispenses dispenser420 ingredient485)
  (clean shaker352)
  (clean shot418)
  (clean shot148)
  (clean shot264)
  (empty shaker352)
  (empty shot418)
  (empty shot148)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker352 l0)
  (shaker-level shaker352 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail192 ingredient485)
  (cocktail-part2 cocktail192 ingredient166)
)
 (:goal
  (and
      (contains shot418 cocktail192)
      (contains shot148 cocktail192)
)))
