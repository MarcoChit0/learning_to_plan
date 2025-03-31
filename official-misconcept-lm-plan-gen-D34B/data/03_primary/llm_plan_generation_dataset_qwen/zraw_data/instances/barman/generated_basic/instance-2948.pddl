(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot239 - shot
      ingredient158 ingredient119 ingredient342 ingredient17 - ingredient
      cocktail213 - cocktail
      dispenser117 dispenser77 dispenser446 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot239)
  (dispenses dispenser117 ingredient158)
  (dispenses dispenser77 ingredient119)
  (dispenses dispenser446 ingredient342)
  (dispenses dispenser407 ingredient17)
  (clean shaker465)
  (clean shot239)
  (empty shaker465)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail213 ingredient158)
  (cocktail-part2 cocktail213 ingredient17)
)
 (:goal
  (and
      (contains shot239 cocktail213)
)))
