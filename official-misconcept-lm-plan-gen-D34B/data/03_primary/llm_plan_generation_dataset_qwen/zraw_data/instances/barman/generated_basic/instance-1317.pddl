(define (problem prob)
 (:domain barman)
 (:objects 
      shaker482 - shaker
      left right - hand
      shot272 - shot
      ingredient421 ingredient142 ingredient135 ingredient110 - ingredient
      cocktail229 - cocktail
      dispenser425 dispenser22 dispenser67 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker482)
  (ontable shot272)
  (dispenses dispenser425 ingredient421)
  (dispenses dispenser22 ingredient142)
  (dispenses dispenser67 ingredient135)
  (dispenses dispenser303 ingredient110)
  (clean shaker482)
  (clean shot272)
  (empty shaker482)
  (empty shot272)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker482 l0)
  (shaker-level shaker482 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail229 ingredient142)
  (cocktail-part2 cocktail229 ingredient135)
)
 (:goal
  (and
      (contains shot272 cocktail229)
)))
