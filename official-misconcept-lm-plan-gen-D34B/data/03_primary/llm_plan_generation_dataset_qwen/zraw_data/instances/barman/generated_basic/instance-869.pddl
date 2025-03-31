(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot33 - shot
      ingredient439 ingredient319 - ingredient
      cocktail1 - cocktail
      dispenser44 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot33)
  (dispenses dispenser44 ingredient439)
  (dispenses dispenser466 ingredient319)
  (clean shaker373)
  (clean shot33)
  (empty shaker373)
  (empty shot33)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient439)
  (cocktail-part2 cocktail1 ingredient319)
)
 (:goal
  (and
      (contains shot33 cocktail1)
)))
