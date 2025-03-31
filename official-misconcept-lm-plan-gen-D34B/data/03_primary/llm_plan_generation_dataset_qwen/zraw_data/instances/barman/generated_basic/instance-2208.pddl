(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot462 shot468 shot116 - shot
      ingredient103 ingredient170 ingredient42 ingredient79 - ingredient
      cocktail362 - cocktail
      dispenser115 dispenser85 dispenser381 dispenser121 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot462)
  (ontable shot468)
  (ontable shot116)
  (dispenses dispenser115 ingredient103)
  (dispenses dispenser85 ingredient170)
  (dispenses dispenser381 ingredient42)
  (dispenses dispenser121 ingredient79)
  (clean shaker264)
  (clean shot462)
  (clean shot468)
  (clean shot116)
  (empty shaker264)
  (empty shot462)
  (empty shot468)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail362 ingredient103)
  (cocktail-part2 cocktail362 ingredient42)
)
 (:goal
  (and
      (contains shot462 cocktail362)
      (contains shot468 ingredient42)
)))
