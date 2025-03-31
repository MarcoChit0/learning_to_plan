(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot336 shot380 - shot
      ingredient421 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser90 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot336)
  (ontable shot380)
  (dispenses dispenser90 ingredient421)
  (dispenses dispenser345 ingredient390)
  (clean shaker180)
  (clean shot336)
  (clean shot380)
  (empty shaker180)
  (empty shot336)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient421)
  (cocktail-part2 cocktail1 ingredient390)
)
 (:goal
  (and
      (contains shot336 cocktail1)
)))
