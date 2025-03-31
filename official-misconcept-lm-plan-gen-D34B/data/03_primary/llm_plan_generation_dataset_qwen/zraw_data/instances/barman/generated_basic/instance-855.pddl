(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot72 - shot
      ingredient467 ingredient111 ingredient437 - ingredient
      cocktail1 - cocktail
      dispenser376 dispenser221 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot72)
  (dispenses dispenser376 ingredient467)
  (dispenses dispenser221 ingredient111)
  (dispenses dispenser118 ingredient437)
  (clean shaker358)
  (clean shot72)
  (empty shaker358)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient111)
  (cocktail-part2 cocktail1 ingredient437)
)
 (:goal
  (and
      (contains shot72 cocktail1)
)))
