(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot199 - shot
      ingredient115 ingredient321 - ingredient
      cocktail69 - cocktail
      dispenser132 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot199)
  (dispenses dispenser132 ingredient115)
  (dispenses dispenser445 ingredient321)
  (clean shaker122)
  (clean shot199)
  (empty shaker122)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient115)
  (cocktail-part2 cocktail69 ingredient321)
)
 (:goal
  (and
      (contains shot199 cocktail69)
)))
