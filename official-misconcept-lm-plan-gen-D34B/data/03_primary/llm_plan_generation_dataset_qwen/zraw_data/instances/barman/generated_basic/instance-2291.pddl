(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot232 - shot
      ingredient190 ingredient187 ingredient486 - ingredient
      cocktail418 - cocktail
      dispenser473 dispenser343 dispenser17 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot232)
  (dispenses dispenser473 ingredient190)
  (dispenses dispenser343 ingredient187)
  (dispenses dispenser17 ingredient486)
  (clean shaker359)
  (clean shot232)
  (empty shaker359)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient187)
  (cocktail-part2 cocktail418 ingredient486)
)
 (:goal
  (and
      (contains shot232 cocktail418)
)))
