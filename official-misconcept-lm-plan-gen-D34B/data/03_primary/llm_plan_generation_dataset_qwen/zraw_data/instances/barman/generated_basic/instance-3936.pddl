(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot132 shot394 - shot
      ingredient361 ingredient470 - ingredient
      cocktail124 - cocktail
      dispenser419 dispenser473 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot132)
  (ontable shot394)
  (dispenses dispenser419 ingredient361)
  (dispenses dispenser473 ingredient470)
  (clean shaker91)
  (clean shot132)
  (clean shot394)
  (empty shaker91)
  (empty shot132)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient470)
  (cocktail-part2 cocktail124 ingredient361)
)
 (:goal
  (and
      (contains shot132 cocktail124)
)))
