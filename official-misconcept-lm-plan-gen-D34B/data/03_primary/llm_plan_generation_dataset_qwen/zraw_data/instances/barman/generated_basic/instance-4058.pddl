(define (problem prob)
 (:domain barman)
 (:objects 
      shaker138 - shaker
      left right - hand
      shot107 shot39 - shot
      ingredient265 ingredient104 - ingredient
      cocktail355 - cocktail
      dispenser153 dispenser233 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker138)
  (ontable shot107)
  (ontable shot39)
  (dispenses dispenser153 ingredient265)
  (dispenses dispenser233 ingredient104)
  (clean shaker138)
  (clean shot107)
  (clean shot39)
  (empty shaker138)
  (empty shot107)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker138 l0)
  (shaker-level shaker138 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient104)
  (cocktail-part2 cocktail355 ingredient265)
)
 (:goal
  (and
      (contains shot107 cocktail355)
)))
