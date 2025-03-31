(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot364 - shot
      ingredient378 ingredient65 ingredient42 ingredient489 - ingredient
      cocktail277 - cocktail
      dispenser155 dispenser282 dispenser420 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot364)
  (dispenses dispenser155 ingredient378)
  (dispenses dispenser282 ingredient65)
  (dispenses dispenser420 ingredient42)
  (dispenses dispenser14 ingredient489)
  (clean shaker273)
  (clean shot364)
  (empty shaker273)
  (empty shot364)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient378)
  (cocktail-part2 cocktail277 ingredient65)
)
 (:goal
  (and
      (contains shot364 cocktail277)
)))
