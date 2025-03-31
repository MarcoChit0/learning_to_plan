(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot43 - shot
      ingredient388 ingredient373 ingredient115 ingredient264 - ingredient
      cocktail335 - cocktail
      dispenser33 dispenser240 dispenser82 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot43)
  (dispenses dispenser33 ingredient388)
  (dispenses dispenser240 ingredient373)
  (dispenses dispenser82 ingredient115)
  (dispenses dispenser449 ingredient264)
  (clean shaker463)
  (clean shot43)
  (empty shaker463)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail335 ingredient373)
  (cocktail-part2 cocktail335 ingredient388)
)
 (:goal
  (and
      (contains shot43 cocktail335)
)))
