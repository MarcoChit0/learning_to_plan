(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot22 - shot
      ingredient295 ingredient368 - ingredient
      cocktail65 - cocktail
      dispenser359 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot22)
  (dispenses dispenser359 ingredient295)
  (dispenses dispenser143 ingredient368)
  (clean shaker460)
  (clean shot22)
  (empty shaker460)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail65 ingredient295)
  (cocktail-part2 cocktail65 ingredient368)
)
 (:goal
  (and
      (contains shot22 cocktail65)
)))
