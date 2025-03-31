(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot335 shot11 - shot
      ingredient15 ingredient20 ingredient374 ingredient48 - ingredient
      cocktail347 - cocktail
      dispenser194 dispenser156 dispenser210 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot335)
  (ontable shot11)
  (dispenses dispenser194 ingredient15)
  (dispenses dispenser156 ingredient20)
  (dispenses dispenser210 ingredient374)
  (dispenses dispenser254 ingredient48)
  (clean shaker227)
  (clean shot335)
  (clean shot11)
  (empty shaker227)
  (empty shot335)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient48)
  (cocktail-part2 cocktail347 ingredient20)
)
 (:goal
  (and
      (contains shot335 cocktail347)
)))
