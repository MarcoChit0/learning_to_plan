(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot373 shot305 shot476 - shot
      ingredient166 ingredient482 ingredient4 ingredient37 - ingredient
      cocktail380 - cocktail
      dispenser347 dispenser179 dispenser131 dispenser469 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot373)
  (ontable shot305)
  (ontable shot476)
  (dispenses dispenser347 ingredient166)
  (dispenses dispenser179 ingredient482)
  (dispenses dispenser131 ingredient4)
  (dispenses dispenser469 ingredient37)
  (clean shaker163)
  (clean shot373)
  (clean shot305)
  (clean shot476)
  (empty shaker163)
  (empty shot373)
  (empty shot305)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient166)
  (cocktail-part2 cocktail380 ingredient4)
)
 (:goal
  (and
      (contains shot373 cocktail380)
      (contains shot305 ingredient482)
)))
