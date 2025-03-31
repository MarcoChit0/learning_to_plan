(define (problem prob)
 (:domain barman)
 (:objects 
      shaker253 - shaker
      left right - hand
      shot81 - shot
      ingredient191 ingredient474 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser471 dispenser343 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker253)
  (ontable shot81)
  (dispenses dispenser471 ingredient191)
  (dispenses dispenser343 ingredient474)
  (dispenses dispenser346 ingredient250)
  (clean shaker253)
  (clean shot81)
  (empty shaker253)
  (empty shot81)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker253 l0)
  (shaker-level shaker253 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient191)
  (cocktail-part2 cocktail1 ingredient474)
)
 (:goal
  (and
      (contains shot81 cocktail1)
)))
