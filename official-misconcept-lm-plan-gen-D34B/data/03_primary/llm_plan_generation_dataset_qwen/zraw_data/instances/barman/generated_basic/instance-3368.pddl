(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot324 - shot
      ingredient220 ingredient260 - ingredient
      cocktail496 - cocktail
      dispenser161 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot324)
  (dispenses dispenser161 ingredient220)
  (dispenses dispenser193 ingredient260)
  (clean shaker163)
  (clean shot324)
  (empty shaker163)
  (empty shot324)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail496 ingredient220)
  (cocktail-part2 cocktail496 ingredient260)
)
 (:goal
  (and
      (contains shot324 cocktail496)
)))
