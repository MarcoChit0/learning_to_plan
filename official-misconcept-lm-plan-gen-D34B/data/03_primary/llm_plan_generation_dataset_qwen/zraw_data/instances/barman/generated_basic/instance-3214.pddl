(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot93 - shot
      ingredient177 ingredient68 - ingredient
      cocktail44 - cocktail
      dispenser75 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot93)
  (dispenses dispenser75 ingredient177)
  (dispenses dispenser469 ingredient68)
  (clean shaker229)
  (clean shot93)
  (empty shaker229)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient177)
  (cocktail-part2 cocktail44 ingredient68)
)
 (:goal
  (and
      (contains shot93 cocktail44)
)))
