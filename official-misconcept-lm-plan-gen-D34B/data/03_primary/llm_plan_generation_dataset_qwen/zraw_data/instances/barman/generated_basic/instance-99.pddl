(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot384 - shot
      ingredient328 ingredient366 ingredient400 ingredient342 - ingredient
      cocktail1 - cocktail
      dispenser407 dispenser52 dispenser151 dispenser158 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot384)
  (dispenses dispenser407 ingredient328)
  (dispenses dispenser52 ingredient366)
  (dispenses dispenser151 ingredient400)
  (dispenses dispenser158 ingredient342)
  (clean shaker303)
  (clean shot384)
  (empty shaker303)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient328)
  (cocktail-part2 cocktail1 ingredient342)
)
 (:goal
  (and
      (contains shot384 cocktail1)
)))
