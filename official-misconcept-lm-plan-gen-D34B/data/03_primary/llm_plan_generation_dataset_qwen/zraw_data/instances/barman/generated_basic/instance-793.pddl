(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot321 - shot
      ingredient7 ingredient356 - ingredient
      cocktail1 - cocktail
      dispenser170 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot321)
  (dispenses dispenser170 ingredient7)
  (dispenses dispenser483 ingredient356)
  (clean shaker171)
  (clean shot321)
  (empty shaker171)
  (empty shot321)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient7)
  (cocktail-part2 cocktail1 ingredient356)
)
 (:goal
  (and
      (contains shot321 cocktail1)
)))
