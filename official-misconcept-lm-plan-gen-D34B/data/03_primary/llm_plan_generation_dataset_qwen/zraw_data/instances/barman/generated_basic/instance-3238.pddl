(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot48 - shot
      ingredient80 ingredient295 ingredient71 - ingredient
      cocktail29 - cocktail
      dispenser42 dispenser18 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot48)
  (dispenses dispenser42 ingredient80)
  (dispenses dispenser18 ingredient295)
  (dispenses dispenser323 ingredient71)
  (clean shaker171)
  (clean shot48)
  (empty shaker171)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient295)
  (cocktail-part2 cocktail29 ingredient71)
)
 (:goal
  (and
      (contains shot48 cocktail29)
)))
