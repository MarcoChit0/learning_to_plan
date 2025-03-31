(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot442 shot189 shot235 - shot
      ingredient19 ingredient159 - ingredient
      cocktail269 - cocktail
      dispenser83 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot442)
  (ontable shot189)
  (ontable shot235)
  (dispenses dispenser83 ingredient19)
  (dispenses dispenser388 ingredient159)
  (clean shaker26)
  (clean shot442)
  (clean shot189)
  (clean shot235)
  (empty shaker26)
  (empty shot442)
  (empty shot189)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient19)
  (cocktail-part2 cocktail269 ingredient159)
)
 (:goal
  (and
      (contains shot442 cocktail269)
      (contains shot189 cocktail269)
)))
