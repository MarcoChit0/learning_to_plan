(define (problem prob)
 (:domain barman)
 (:objects 
      shaker363 - shaker
      left right - hand
      shot133 - shot
      ingredient0 ingredient405 ingredient125 - ingredient
      cocktail1 - cocktail
      dispenser161 dispenser162 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker363)
  (ontable shot133)
  (dispenses dispenser161 ingredient0)
  (dispenses dispenser162 ingredient405)
  (dispenses dispenser68 ingredient125)
  (clean shaker363)
  (clean shot133)
  (empty shaker363)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker363 l0)
  (shaker-level shaker363 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient405)
  (cocktail-part2 cocktail1 ingredient125)
)
 (:goal
  (and
      (contains shot133 cocktail1)
)))
