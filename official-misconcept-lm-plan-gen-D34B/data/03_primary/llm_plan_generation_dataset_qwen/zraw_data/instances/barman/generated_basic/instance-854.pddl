(define (problem prob)
 (:domain barman)
 (:objects 
      shaker255 - shaker
      left right - hand
      shot226 - shot
      ingredient308 ingredient136 ingredient168 - ingredient
      cocktail1 - cocktail
      dispenser176 dispenser362 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker255)
  (ontable shot226)
  (dispenses dispenser176 ingredient308)
  (dispenses dispenser362 ingredient136)
  (dispenses dispenser44 ingredient168)
  (clean shaker255)
  (clean shot226)
  (empty shaker255)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker255 l0)
  (shaker-level shaker255 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient136)
  (cocktail-part2 cocktail1 ingredient168)
)
 (:goal
  (and
      (contains shot226 cocktail1)
)))
