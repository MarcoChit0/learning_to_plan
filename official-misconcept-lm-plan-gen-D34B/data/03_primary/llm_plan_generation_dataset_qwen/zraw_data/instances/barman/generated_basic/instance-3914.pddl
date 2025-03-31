(define (problem prob)
 (:domain barman)
 (:objects 
      shaker206 - shaker
      left right - hand
      shot493 - shot
      ingredient463 ingredient17 ingredient386 ingredient272 - ingredient
      cocktail367 - cocktail
      dispenser282 dispenser39 dispenser68 dispenser298 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker206)
  (ontable shot493)
  (dispenses dispenser282 ingredient463)
  (dispenses dispenser39 ingredient17)
  (dispenses dispenser68 ingredient386)
  (dispenses dispenser298 ingredient272)
  (clean shaker206)
  (clean shot493)
  (empty shaker206)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker206 l0)
  (shaker-level shaker206 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient272)
  (cocktail-part2 cocktail367 ingredient17)
)
 (:goal
  (and
      (contains shot493 cocktail367)
)))
