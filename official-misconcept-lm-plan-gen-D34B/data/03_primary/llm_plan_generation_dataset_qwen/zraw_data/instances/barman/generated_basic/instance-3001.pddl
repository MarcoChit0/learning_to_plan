(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot145 shot442 - shot
      ingredient137 ingredient87 ingredient489 - ingredient
      cocktail453 - cocktail
      dispenser41 dispenser122 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot145)
  (ontable shot442)
  (dispenses dispenser41 ingredient137)
  (dispenses dispenser122 ingredient87)
  (dispenses dispenser223 ingredient489)
  (clean shaker233)
  (clean shot145)
  (clean shot442)
  (empty shaker233)
  (empty shot145)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient87)
  (cocktail-part2 cocktail453 ingredient137)
)
 (:goal
  (and
      (contains shot145 cocktail453)
)))
