(define (problem prob)
 (:domain barman)
 (:objects 
      shaker274 - shaker
      left right - hand
      shot285 - shot
      ingredient117 ingredient446 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker274)
  (ontable shot285)
  (dispenses dispenser42 ingredient117)
  (dispenses dispenser245 ingredient446)
  (clean shaker274)
  (clean shot285)
  (empty shaker274)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker274 l0)
  (shaker-level shaker274 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient117)
  (cocktail-part2 cocktail1 ingredient446)
)
 (:goal
  (and
      (contains shot285 cocktail1)
)))
