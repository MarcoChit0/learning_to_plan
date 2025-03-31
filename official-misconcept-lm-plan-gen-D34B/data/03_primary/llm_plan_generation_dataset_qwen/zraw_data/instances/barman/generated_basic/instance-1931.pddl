(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot117 - shot
      ingredient498 ingredient425 - ingredient
      cocktail164 - cocktail
      dispenser457 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot117)
  (dispenses dispenser457 ingredient498)
  (dispenses dispenser390 ingredient425)
  (clean shaker307)
  (clean shot117)
  (empty shaker307)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient425)
  (cocktail-part2 cocktail164 ingredient498)
)
 (:goal
  (and
      (contains shot117 cocktail164)
)))
