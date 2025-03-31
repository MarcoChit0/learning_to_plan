(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot493 - shot
      ingredient354 ingredient488 - ingredient
      cocktail1 - cocktail
      dispenser44 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot493)
  (dispenses dispenser44 ingredient354)
  (dispenses dispenser276 ingredient488)
  (clean shaker188)
  (clean shot493)
  (empty shaker188)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient488)
  (cocktail-part2 cocktail1 ingredient354)
)
 (:goal
  (and
      (contains shot493 cocktail1)
)))
