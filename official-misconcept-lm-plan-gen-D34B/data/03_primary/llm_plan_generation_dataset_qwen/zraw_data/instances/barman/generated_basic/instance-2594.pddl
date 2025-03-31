(define (problem prob)
 (:domain barman)
 (:objects 
      shaker84 - shaker
      left right - hand
      shot448 shot179 - shot
      ingredient218 ingredient406 ingredient269 - ingredient
      cocktail250 - cocktail
      dispenser62 dispenser76 dispenser318 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker84)
  (ontable shot448)
  (ontable shot179)
  (dispenses dispenser62 ingredient218)
  (dispenses dispenser76 ingredient406)
  (dispenses dispenser318 ingredient269)
  (clean shaker84)
  (clean shot448)
  (clean shot179)
  (empty shaker84)
  (empty shot448)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker84 l0)
  (shaker-level shaker84 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient406)
  (cocktail-part2 cocktail250 ingredient269)
)
 (:goal
  (and
      (contains shot448 cocktail250)
)))
