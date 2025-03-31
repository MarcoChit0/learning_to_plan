(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot372 - shot
      ingredient464 ingredient337 - ingredient
      cocktail367 - cocktail
      dispenser221 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot372)
  (dispenses dispenser221 ingredient464)
  (dispenses dispenser34 ingredient337)
  (clean shaker426)
  (clean shot372)
  (empty shaker426)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient337)
  (cocktail-part2 cocktail367 ingredient464)
)
 (:goal
  (and
      (contains shot372 cocktail367)
)))
