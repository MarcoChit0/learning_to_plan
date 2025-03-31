(define (problem prob)
 (:domain barman)
 (:objects 
      shaker367 - shaker
      left right - hand
      shot132 shot129 - shot
      ingredient434 ingredient223 - ingredient
      cocktail26 - cocktail
      dispenser285 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker367)
  (ontable shot132)
  (ontable shot129)
  (dispenses dispenser285 ingredient434)
  (dispenses dispenser157 ingredient223)
  (clean shaker367)
  (clean shot132)
  (clean shot129)
  (empty shaker367)
  (empty shot132)
  (empty shot129)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker367 l0)
  (shaker-level shaker367 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail26 ingredient223)
  (cocktail-part2 cocktail26 ingredient434)
)
 (:goal
  (and
      (contains shot132 cocktail26)
)))
