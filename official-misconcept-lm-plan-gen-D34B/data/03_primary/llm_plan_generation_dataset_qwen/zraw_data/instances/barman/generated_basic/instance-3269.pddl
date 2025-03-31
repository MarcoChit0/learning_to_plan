(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot264 shot401 - shot
      ingredient414 ingredient31 ingredient386 - ingredient
      cocktail80 - cocktail
      dispenser270 dispenser357 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot264)
  (ontable shot401)
  (dispenses dispenser270 ingredient414)
  (dispenses dispenser357 ingredient31)
  (dispenses dispenser273 ingredient386)
  (clean shaker411)
  (clean shot264)
  (clean shot401)
  (empty shaker411)
  (empty shot264)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient386)
  (cocktail-part2 cocktail80 ingredient31)
)
 (:goal
  (and
      (contains shot264 cocktail80)
)))
