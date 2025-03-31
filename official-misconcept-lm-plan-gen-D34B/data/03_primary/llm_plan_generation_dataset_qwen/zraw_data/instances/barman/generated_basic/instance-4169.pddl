(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot249 shot131 - shot
      ingredient107 ingredient200 - ingredient
      cocktail157 - cocktail
      dispenser435 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot249)
  (ontable shot131)
  (dispenses dispenser435 ingredient107)
  (dispenses dispenser432 ingredient200)
  (clean shaker216)
  (clean shot249)
  (clean shot131)
  (empty shaker216)
  (empty shot249)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail157 ingredient200)
  (cocktail-part2 cocktail157 ingredient107)
)
 (:goal
  (and
      (contains shot249 cocktail157)
)))
