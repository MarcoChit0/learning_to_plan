(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot332 - shot
      ingredient80 ingredient204 - ingredient
      cocktail207 - cocktail
      dispenser252 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot332)
  (dispenses dispenser252 ingredient80)
  (dispenses dispenser357 ingredient204)
  (clean shaker211)
  (clean shot332)
  (empty shaker211)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient80)
  (cocktail-part2 cocktail207 ingredient204)
)
 (:goal
  (and
      (contains shot332 cocktail207)
)))
