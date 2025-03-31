(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot152 - shot
      ingredient63 ingredient126 ingredient400 ingredient318 - ingredient
      cocktail291 - cocktail
      dispenser256 dispenser434 dispenser23 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot152)
  (dispenses dispenser256 ingredient63)
  (dispenses dispenser434 ingredient126)
  (dispenses dispenser23 ingredient400)
  (dispenses dispenser124 ingredient318)
  (clean shaker479)
  (clean shot152)
  (empty shaker479)
  (empty shot152)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient400)
  (cocktail-part2 cocktail291 ingredient63)
)
 (:goal
  (and
      (contains shot152 cocktail291)
)))
