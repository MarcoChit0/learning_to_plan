(define (problem prob)
 (:domain barman)
 (:objects 
      shaker495 - shaker
      left right - hand
      shot369 shot15 shot465 - shot
      ingredient414 ingredient382 ingredient493 - ingredient
      cocktail1 - cocktail
      dispenser281 dispenser365 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker495)
  (ontable shot369)
  (ontable shot15)
  (ontable shot465)
  (dispenses dispenser281 ingredient414)
  (dispenses dispenser365 ingredient382)
  (dispenses dispenser340 ingredient493)
  (clean shaker495)
  (clean shot369)
  (clean shot15)
  (clean shot465)
  (empty shaker495)
  (empty shot369)
  (empty shot15)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker495 l0)
  (shaker-level shaker495 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient493)
  (cocktail-part2 cocktail1 ingredient414)
)
 (:goal
  (and
      (contains shot369 cocktail1)
      (contains shot15 ingredient493)
)))
