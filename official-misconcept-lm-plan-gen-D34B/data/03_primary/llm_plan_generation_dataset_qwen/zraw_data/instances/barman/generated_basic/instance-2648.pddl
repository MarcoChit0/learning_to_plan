(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot448 shot356 - shot
      ingredient361 ingredient119 ingredient184 - ingredient
      cocktail29 - cocktail
      dispenser64 dispenser350 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot448)
  (ontable shot356)
  (dispenses dispenser64 ingredient361)
  (dispenses dispenser350 ingredient119)
  (dispenses dispenser180 ingredient184)
  (clean shaker122)
  (clean shot448)
  (clean shot356)
  (empty shaker122)
  (empty shot448)
  (empty shot356)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient361)
  (cocktail-part2 cocktail29 ingredient184)
)
 (:goal
  (and
      (contains shot448 cocktail29)
)))
