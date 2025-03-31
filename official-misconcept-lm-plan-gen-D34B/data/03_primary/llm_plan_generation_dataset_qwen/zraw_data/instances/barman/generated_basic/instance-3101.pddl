(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot417 - shot
      ingredient235 ingredient13 ingredient304 ingredient111 - ingredient
      cocktail264 - cocktail
      dispenser14 dispenser202 dispenser256 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot417)
  (dispenses dispenser14 ingredient235)
  (dispenses dispenser202 ingredient13)
  (dispenses dispenser256 ingredient304)
  (dispenses dispenser388 ingredient111)
  (clean shaker412)
  (clean shot417)
  (empty shaker412)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail264 ingredient13)
  (cocktail-part2 cocktail264 ingredient235)
)
 (:goal
  (and
      (contains shot417 cocktail264)
)))
