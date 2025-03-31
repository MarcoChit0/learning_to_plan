(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot189 shot473 - shot
      ingredient499 ingredient216 - ingredient
      cocktail162 - cocktail
      dispenser103 dispenser439 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot189)
  (ontable shot473)
  (dispenses dispenser103 ingredient499)
  (dispenses dispenser439 ingredient216)
  (clean shaker62)
  (clean shot189)
  (clean shot473)
  (empty shaker62)
  (empty shot189)
  (empty shot473)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient216)
  (cocktail-part2 cocktail162 ingredient499)
)
 (:goal
  (and
      (contains shot189 cocktail162)
)))
