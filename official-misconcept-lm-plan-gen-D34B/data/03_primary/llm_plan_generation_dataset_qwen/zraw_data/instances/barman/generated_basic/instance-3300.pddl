(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot145 - shot
      ingredient337 ingredient33 - ingredient
      cocktail227 - cocktail
      dispenser132 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot145)
  (dispenses dispenser132 ingredient337)
  (dispenses dispenser352 ingredient33)
  (clean shaker38)
  (clean shot145)
  (empty shaker38)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient33)
  (cocktail-part2 cocktail227 ingredient337)
)
 (:goal
  (and
      (contains shot145 cocktail227)
)))
