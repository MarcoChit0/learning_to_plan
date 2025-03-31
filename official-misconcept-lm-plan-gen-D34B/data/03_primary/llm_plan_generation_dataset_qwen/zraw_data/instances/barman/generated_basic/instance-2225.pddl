(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot335 - shot
      ingredient400 ingredient416 - ingredient
      cocktail89 - cocktail
      dispenser273 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot335)
  (dispenses dispenser273 ingredient400)
  (dispenses dispenser482 ingredient416)
  (clean shaker45)
  (clean shot335)
  (empty shaker45)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail89 ingredient416)
  (cocktail-part2 cocktail89 ingredient400)
)
 (:goal
  (and
      (contains shot335 cocktail89)
)))
