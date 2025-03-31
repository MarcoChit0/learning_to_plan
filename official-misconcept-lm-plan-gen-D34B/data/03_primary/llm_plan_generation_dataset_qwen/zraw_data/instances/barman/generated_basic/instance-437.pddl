(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot479 shot417 - shot
      ingredient264 ingredient485 ingredient406 - ingredient
      cocktail1 - cocktail
      dispenser484 dispenser444 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot479)
  (ontable shot417)
  (dispenses dispenser484 ingredient264)
  (dispenses dispenser444 ingredient485)
  (dispenses dispenser493 ingredient406)
  (clean shaker247)
  (clean shot479)
  (clean shot417)
  (empty shaker247)
  (empty shot479)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient264)
  (cocktail-part2 cocktail1 ingredient485)
)
 (:goal
  (and
      (contains shot479 cocktail1)
)))
