(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot201 - shot
      ingredient21 ingredient378 - ingredient
      cocktail251 - cocktail
      dispenser313 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot201)
  (dispenses dispenser313 ingredient21)
  (dispenses dispenser212 ingredient378)
  (clean shaker479)
  (clean shot201)
  (empty shaker479)
  (empty shot201)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient21)
  (cocktail-part2 cocktail251 ingredient378)
)
 (:goal
  (and
      (contains shot201 cocktail251)
)))
