(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot463 - shot
      ingredient147 ingredient330 ingredient454 ingredient195 - ingredient
      cocktail472 - cocktail
      dispenser318 dispenser52 dispenser38 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot463)
  (dispenses dispenser318 ingredient147)
  (dispenses dispenser52 ingredient330)
  (dispenses dispenser38 ingredient454)
  (dispenses dispenser263 ingredient195)
  (clean shaker359)
  (clean shot463)
  (empty shaker359)
  (empty shot463)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail472 ingredient330)
  (cocktail-part2 cocktail472 ingredient195)
)
 (:goal
  (and
      (contains shot463 cocktail472)
)))
