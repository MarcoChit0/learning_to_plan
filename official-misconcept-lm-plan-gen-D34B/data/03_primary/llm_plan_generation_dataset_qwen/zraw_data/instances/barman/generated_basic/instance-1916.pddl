(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot23 - shot
      ingredient227 ingredient158 ingredient338 ingredient283 - ingredient
      cocktail325 - cocktail
      dispenser248 dispenser334 dispenser313 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot23)
  (dispenses dispenser248 ingredient227)
  (dispenses dispenser334 ingredient158)
  (dispenses dispenser313 ingredient338)
  (dispenses dispenser388 ingredient283)
  (clean shaker107)
  (clean shot23)
  (empty shaker107)
  (empty shot23)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail325 ingredient227)
  (cocktail-part2 cocktail325 ingredient283)
)
 (:goal
  (and
      (contains shot23 cocktail325)
)))
