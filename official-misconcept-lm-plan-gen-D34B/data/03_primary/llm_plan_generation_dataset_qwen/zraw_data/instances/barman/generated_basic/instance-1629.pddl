(define (problem prob)
 (:domain barman)
 (:objects 
      shaker354 - shaker
      left right - hand
      shot490 shot354 shot190 - shot
      ingredient237 ingredient479 - ingredient
      cocktail359 - cocktail
      dispenser46 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker354)
  (ontable shot490)
  (ontable shot354)
  (ontable shot190)
  (dispenses dispenser46 ingredient237)
  (dispenses dispenser290 ingredient479)
  (clean shaker354)
  (clean shot490)
  (clean shot354)
  (clean shot190)
  (empty shaker354)
  (empty shot490)
  (empty shot354)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker354 l0)
  (shaker-level shaker354 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail359 ingredient237)
  (cocktail-part2 cocktail359 ingredient479)
)
 (:goal
  (and
      (contains shot490 cocktail359)
      (contains shot354 cocktail359)
)))
