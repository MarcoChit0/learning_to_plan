(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot220 - shot
      ingredient166 ingredient31 ingredient439 ingredient345 - ingredient
      cocktail74 - cocktail
      dispenser162 dispenser157 dispenser489 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot220)
  (dispenses dispenser162 ingredient166)
  (dispenses dispenser157 ingredient31)
  (dispenses dispenser489 ingredient439)
  (dispenses dispenser130 ingredient345)
  (clean shaker319)
  (clean shot220)
  (empty shaker319)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient166)
  (cocktail-part2 cocktail74 ingredient439)
)
 (:goal
  (and
      (contains shot220 cocktail74)
)))
