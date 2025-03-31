(define (problem prob)
 (:domain barman)
 (:objects 
      shaker40 - shaker
      left right - hand
      shot168 shot242 shot33 - shot
      ingredient156 ingredient147 ingredient121 - ingredient
      cocktail44 - cocktail
      dispenser63 dispenser448 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker40)
  (ontable shot168)
  (ontable shot242)
  (ontable shot33)
  (dispenses dispenser63 ingredient156)
  (dispenses dispenser448 ingredient147)
  (dispenses dispenser43 ingredient121)
  (clean shaker40)
  (clean shot168)
  (clean shot242)
  (clean shot33)
  (empty shaker40)
  (empty shot168)
  (empty shot242)
  (empty shot33)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker40 l0)
  (shaker-level shaker40 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient156)
  (cocktail-part2 cocktail44 ingredient121)
)
 (:goal
  (and
      (contains shot168 cocktail44)
      (contains shot242 ingredient121)
)))
