(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot113 - shot
      ingredient96 ingredient474 ingredient264 - ingredient
      cocktail1 - cocktail
      dispenser484 dispenser324 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot113)
  (dispenses dispenser484 ingredient96)
  (dispenses dispenser324 ingredient474)
  (dispenses dispenser128 ingredient264)
  (clean shaker315)
  (clean shot113)
  (empty shaker315)
  (empty shot113)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient474)
  (cocktail-part2 cocktail1 ingredient96)
)
 (:goal
  (and
      (contains shot113 cocktail1)
)))
