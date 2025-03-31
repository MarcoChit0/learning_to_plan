(define (problem prob)
 (:domain barman)
 (:objects 
      shaker133 - shaker
      left right - hand
      shot193 shot186 - shot
      ingredient19 ingredient247 ingredient308 ingredient32 - ingredient
      cocktail1 - cocktail
      dispenser23 dispenser392 dispenser48 dispenser312 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker133)
  (ontable shot193)
  (ontable shot186)
  (dispenses dispenser23 ingredient19)
  (dispenses dispenser392 ingredient247)
  (dispenses dispenser48 ingredient308)
  (dispenses dispenser312 ingredient32)
  (clean shaker133)
  (clean shot193)
  (clean shot186)
  (empty shaker133)
  (empty shot193)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker133 l0)
  (shaker-level shaker133 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient247)
  (cocktail-part2 cocktail1 ingredient308)
)
 (:goal
  (and
      (contains shot193 cocktail1)
)))
