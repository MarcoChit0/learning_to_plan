(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot424 shot150 - shot
      ingredient43 ingredient413 ingredient233 - ingredient
      cocktail233 - cocktail
      dispenser337 dispenser293 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot424)
  (ontable shot150)
  (dispenses dispenser337 ingredient43)
  (dispenses dispenser293 ingredient413)
  (dispenses dispenser305 ingredient233)
  (clean shaker359)
  (clean shot424)
  (clean shot150)
  (empty shaker359)
  (empty shot424)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail233 ingredient43)
  (cocktail-part2 cocktail233 ingredient233)
)
 (:goal
  (and
      (contains shot424 cocktail233)
)))
