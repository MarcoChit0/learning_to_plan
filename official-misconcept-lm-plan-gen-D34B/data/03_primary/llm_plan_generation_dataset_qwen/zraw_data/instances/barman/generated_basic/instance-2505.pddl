(define (problem prob)
 (:domain barman)
 (:objects 
      shaker81 - shaker
      left right - hand
      shot14 - shot
      ingredient165 ingredient460 - ingredient
      cocktail195 - cocktail
      dispenser248 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker81)
  (ontable shot14)
  (dispenses dispenser248 ingredient165)
  (dispenses dispenser73 ingredient460)
  (clean shaker81)
  (clean shot14)
  (empty shaker81)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker81 l0)
  (shaker-level shaker81 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient460)
  (cocktail-part2 cocktail195 ingredient165)
)
 (:goal
  (and
      (contains shot14 cocktail195)
)))
