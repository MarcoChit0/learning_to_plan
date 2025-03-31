(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot295 - shot
      ingredient366 ingredient64 - ingredient
      cocktail1 - cocktail
      dispenser430 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot295)
  (dispenses dispenser430 ingredient366)
  (dispenses dispenser355 ingredient64)
  (clean shaker479)
  (clean shot295)
  (empty shaker479)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient64)
  (cocktail-part2 cocktail1 ingredient366)
)
 (:goal
  (and
      (contains shot295 cocktail1)
)))
