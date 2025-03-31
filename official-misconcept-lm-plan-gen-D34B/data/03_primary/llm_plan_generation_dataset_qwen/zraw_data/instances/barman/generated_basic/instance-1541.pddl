(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot396 - shot
      ingredient391 ingredient31 - ingredient
      cocktail134 - cocktail
      dispenser62 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot396)
  (dispenses dispenser62 ingredient391)
  (dispenses dispenser391 ingredient31)
  (clean shaker377)
  (clean shot396)
  (empty shaker377)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail134 ingredient391)
  (cocktail-part2 cocktail134 ingredient31)
)
 (:goal
  (and
      (contains shot396 cocktail134)
)))
