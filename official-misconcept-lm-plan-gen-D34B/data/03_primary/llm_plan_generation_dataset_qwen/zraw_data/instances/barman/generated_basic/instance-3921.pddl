(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot229 - shot
      ingredient156 ingredient223 ingredient44 ingredient72 - ingredient
      cocktail217 - cocktail
      dispenser30 dispenser60 dispenser179 dispenser481 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot229)
  (dispenses dispenser30 ingredient156)
  (dispenses dispenser60 ingredient223)
  (dispenses dispenser179 ingredient44)
  (dispenses dispenser481 ingredient72)
  (clean shaker328)
  (clean shot229)
  (empty shaker328)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail217 ingredient72)
  (cocktail-part2 cocktail217 ingredient223)
)
 (:goal
  (and
      (contains shot229 cocktail217)
)))
