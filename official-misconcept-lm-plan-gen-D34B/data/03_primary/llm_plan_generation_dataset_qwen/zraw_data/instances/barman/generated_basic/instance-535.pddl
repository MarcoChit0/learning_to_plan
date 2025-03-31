(define (problem prob)
 (:domain barman)
 (:objects 
      shaker97 - shaker
      left right - hand
      shot171 - shot
      ingredient302 ingredient255 ingredient87 - ingredient
      cocktail1 - cocktail
      dispenser62 dispenser76 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker97)
  (ontable shot171)
  (dispenses dispenser62 ingredient302)
  (dispenses dispenser76 ingredient255)
  (dispenses dispenser68 ingredient87)
  (clean shaker97)
  (clean shot171)
  (empty shaker97)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker97 l0)
  (shaker-level shaker97 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient302)
  (cocktail-part2 cocktail1 ingredient87)
)
 (:goal
  (and
      (contains shot171 cocktail1)
)))
