(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot438 shot72 shot62 - shot
      ingredient204 ingredient129 ingredient261 - ingredient
      cocktail272 - cocktail
      dispenser306 dispenser445 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot438)
  (ontable shot72)
  (ontable shot62)
  (dispenses dispenser306 ingredient204)
  (dispenses dispenser445 ingredient129)
  (dispenses dispenser286 ingredient261)
  (clean shaker34)
  (clean shot438)
  (clean shot72)
  (clean shot62)
  (empty shaker34)
  (empty shot438)
  (empty shot72)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail272 ingredient261)
  (cocktail-part2 cocktail272 ingredient129)
)
 (:goal
  (and
      (contains shot438 cocktail272)
      (contains shot72 cocktail272)
)))
