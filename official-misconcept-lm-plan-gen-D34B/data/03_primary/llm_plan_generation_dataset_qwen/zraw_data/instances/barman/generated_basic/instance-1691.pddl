(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot352 shot170 shot254 shot141 - shot
      ingredient327 ingredient421 ingredient352 - ingredient
      cocktail2 - cocktail
      dispenser93 dispenser194 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot352)
  (ontable shot170)
  (ontable shot254)
  (ontable shot141)
  (dispenses dispenser93 ingredient327)
  (dispenses dispenser194 ingredient421)
  (dispenses dispenser402 ingredient352)
  (clean shaker488)
  (clean shot352)
  (clean shot170)
  (clean shot254)
  (clean shot141)
  (empty shaker488)
  (empty shot352)
  (empty shot170)
  (empty shot254)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient327)
  (cocktail-part2 cocktail2 ingredient421)
)
 (:goal
  (and
      (contains shot352 cocktail2)
      (contains shot170 ingredient421)
      (contains shot254 cocktail2)
)))
