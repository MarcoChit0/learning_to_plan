(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot93 - shot
      ingredient148 ingredient479 ingredient122 ingredient372 - ingredient
      cocktail184 - cocktail
      dispenser46 dispenser146 dispenser259 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot93)
  (dispenses dispenser46 ingredient148)
  (dispenses dispenser146 ingredient479)
  (dispenses dispenser259 ingredient122)
  (dispenses dispenser61 ingredient372)
  (clean shaker468)
  (clean shot93)
  (empty shaker468)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient122)
  (cocktail-part2 cocktail184 ingredient372)
)
 (:goal
  (and
      (contains shot93 cocktail184)
)))
