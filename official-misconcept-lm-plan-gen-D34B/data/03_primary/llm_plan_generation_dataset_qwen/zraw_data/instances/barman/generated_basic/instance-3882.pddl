(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot292 shot380 - shot
      ingredient87 ingredient391 ingredient140 - ingredient
      cocktail392 - cocktail
      dispenser36 dispenser466 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot292)
  (ontable shot380)
  (dispenses dispenser36 ingredient87)
  (dispenses dispenser466 ingredient391)
  (dispenses dispenser449 ingredient140)
  (clean shaker359)
  (clean shot292)
  (clean shot380)
  (empty shaker359)
  (empty shot292)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail392 ingredient140)
  (cocktail-part2 cocktail392 ingredient391)
)
 (:goal
  (and
      (contains shot292 cocktail392)
)))
