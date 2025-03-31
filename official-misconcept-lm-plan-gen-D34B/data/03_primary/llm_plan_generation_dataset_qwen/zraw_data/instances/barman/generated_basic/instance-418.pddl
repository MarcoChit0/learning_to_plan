(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot160 - shot
      ingredient103 ingredient425 - ingredient
      cocktail1 - cocktail
      dispenser94 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot160)
  (dispenses dispenser94 ingredient103)
  (dispenses dispenser276 ingredient425)
  (clean shaker163)
  (clean shot160)
  (empty shaker163)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient425)
  (cocktail-part2 cocktail1 ingredient103)
)
 (:goal
  (and
      (contains shot160 cocktail1)
)))
