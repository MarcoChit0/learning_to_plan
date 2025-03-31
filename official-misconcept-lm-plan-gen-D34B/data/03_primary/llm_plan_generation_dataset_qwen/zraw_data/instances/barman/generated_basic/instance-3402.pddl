(define (problem prob)
 (:domain barman)
 (:objects 
      shaker283 - shaker
      left right - hand
      shot199 shot251 - shot
      ingredient85 ingredient372 ingredient469 ingredient217 - ingredient
      cocktail42 - cocktail
      dispenser42 dispenser448 dispenser212 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker283)
  (ontable shot199)
  (ontable shot251)
  (dispenses dispenser42 ingredient85)
  (dispenses dispenser448 ingredient372)
  (dispenses dispenser212 ingredient469)
  (dispenses dispenser269 ingredient217)
  (clean shaker283)
  (clean shot199)
  (clean shot251)
  (empty shaker283)
  (empty shot199)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker283 l0)
  (shaker-level shaker283 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail42 ingredient469)
  (cocktail-part2 cocktail42 ingredient372)
)
 (:goal
  (and
      (contains shot199 cocktail42)
)))
