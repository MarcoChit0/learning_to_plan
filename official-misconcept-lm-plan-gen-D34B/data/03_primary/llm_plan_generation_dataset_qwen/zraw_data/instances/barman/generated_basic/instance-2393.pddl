(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot373 shot350 - shot
      ingredient397 ingredient181 ingredient90 - ingredient
      cocktail13 - cocktail
      dispenser458 dispenser447 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot373)
  (ontable shot350)
  (dispenses dispenser458 ingredient397)
  (dispenses dispenser447 ingredient181)
  (dispenses dispenser284 ingredient90)
  (clean shaker216)
  (clean shot373)
  (clean shot350)
  (empty shaker216)
  (empty shot373)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient181)
  (cocktail-part2 cocktail13 ingredient397)
)
 (:goal
  (and
      (contains shot373 cocktail13)
)))
