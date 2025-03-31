(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot389 shot128 - shot
      ingredient122 ingredient11 ingredient81 - ingredient
      cocktail1 - cocktail
      dispenser213 dispenser362 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot389)
  (ontable shot128)
  (dispenses dispenser213 ingredient122)
  (dispenses dispenser362 ingredient11)
  (dispenses dispenser89 ingredient81)
  (clean shaker356)
  (clean shot389)
  (clean shot128)
  (empty shaker356)
  (empty shot389)
  (empty shot128)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient11)
  (cocktail-part2 cocktail1 ingredient122)
)
 (:goal
  (and
      (contains shot389 cocktail1)
)))
