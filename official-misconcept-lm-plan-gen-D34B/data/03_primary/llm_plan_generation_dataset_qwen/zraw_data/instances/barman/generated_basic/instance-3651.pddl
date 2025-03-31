(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot422 shot389 - shot
      ingredient317 ingredient349 ingredient123 ingredient136 - ingredient
      cocktail390 - cocktail
      dispenser89 dispenser337 dispenser425 dispenser353 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot422)
  (ontable shot389)
  (dispenses dispenser89 ingredient317)
  (dispenses dispenser337 ingredient349)
  (dispenses dispenser425 ingredient123)
  (dispenses dispenser353 ingredient136)
  (clean shaker323)
  (clean shot422)
  (clean shot389)
  (empty shaker323)
  (empty shot422)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient136)
  (cocktail-part2 cocktail390 ingredient349)
)
 (:goal
  (and
      (contains shot422 cocktail390)
)))
