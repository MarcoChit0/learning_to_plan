(define (problem prob)
 (:domain barman)
 (:objects 
      shaker203 - shaker
      left right - hand
      shot156 - shot
      ingredient369 ingredient438 ingredient437 - ingredient
      cocktail162 - cocktail
      dispenser413 dispenser436 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker203)
  (ontable shot156)
  (dispenses dispenser413 ingredient369)
  (dispenses dispenser436 ingredient438)
  (dispenses dispenser434 ingredient437)
  (clean shaker203)
  (clean shot156)
  (empty shaker203)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker203 l0)
  (shaker-level shaker203 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient438)
  (cocktail-part2 cocktail162 ingredient369)
)
 (:goal
  (and
      (contains shot156 cocktail162)
)))
