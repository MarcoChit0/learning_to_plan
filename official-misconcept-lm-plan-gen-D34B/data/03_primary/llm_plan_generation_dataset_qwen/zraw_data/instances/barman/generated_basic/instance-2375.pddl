(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot27 - shot
      ingredient323 ingredient86 ingredient34 ingredient483 - ingredient
      cocktail271 - cocktail
      dispenser209 dispenser91 dispenser101 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot27)
  (dispenses dispenser209 ingredient323)
  (dispenses dispenser91 ingredient86)
  (dispenses dispenser101 ingredient34)
  (dispenses dispenser369 ingredient483)
  (clean shaker236)
  (clean shot27)
  (empty shaker236)
  (empty shot27)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient86)
  (cocktail-part2 cocktail271 ingredient323)
)
 (:goal
  (and
      (contains shot27 cocktail271)
)))
