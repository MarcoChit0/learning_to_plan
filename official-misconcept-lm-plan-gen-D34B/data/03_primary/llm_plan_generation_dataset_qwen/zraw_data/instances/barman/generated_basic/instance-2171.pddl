(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot175 shot193 - shot
      ingredient322 ingredient146 - ingredient
      cocktail319 - cocktail
      dispenser165 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot175)
  (ontable shot193)
  (dispenses dispenser165 ingredient322)
  (dispenses dispenser493 ingredient146)
  (clean shaker172)
  (clean shot175)
  (clean shot193)
  (empty shaker172)
  (empty shot175)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient322)
  (cocktail-part2 cocktail319 ingredient146)
)
 (:goal
  (and
      (contains shot175 cocktail319)
)))
