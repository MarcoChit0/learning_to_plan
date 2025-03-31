(define (problem prob)
 (:domain barman)
 (:objects 
      shaker430 - shaker
      left right - hand
      shot55 - shot
      ingredient129 ingredient115 ingredient328 - ingredient
      cocktail155 - cocktail
      dispenser391 dispenser225 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker430)
  (ontable shot55)
  (dispenses dispenser391 ingredient129)
  (dispenses dispenser225 ingredient115)
  (dispenses dispenser399 ingredient328)
  (clean shaker430)
  (clean shot55)
  (empty shaker430)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker430 l0)
  (shaker-level shaker430 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient129)
  (cocktail-part2 cocktail155 ingredient115)
)
 (:goal
  (and
      (contains shot55 cocktail155)
)))
