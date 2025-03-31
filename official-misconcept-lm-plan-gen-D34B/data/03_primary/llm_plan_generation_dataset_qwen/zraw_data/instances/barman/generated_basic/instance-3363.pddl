(define (problem prob)
 (:domain barman)
 (:objects 
      shaker66 - shaker
      left right - hand
      shot91 - shot
      ingredient369 ingredient112 - ingredient
      cocktail413 - cocktail
      dispenser172 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker66)
  (ontable shot91)
  (dispenses dispenser172 ingredient369)
  (dispenses dispenser361 ingredient112)
  (clean shaker66)
  (clean shot91)
  (empty shaker66)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker66 l0)
  (shaker-level shaker66 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail413 ingredient112)
  (cocktail-part2 cocktail413 ingredient369)
)
 (:goal
  (and
      (contains shot91 cocktail413)
)))
