(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot2 shot487 shot253 - shot
      ingredient154 ingredient489 ingredient348 ingredient376 - ingredient
      cocktail333 - cocktail
      dispenser238 dispenser95 dispenser310 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot2)
  (ontable shot487)
  (ontable shot253)
  (dispenses dispenser238 ingredient154)
  (dispenses dispenser95 ingredient489)
  (dispenses dispenser310 ingredient348)
  (dispenses dispenser465 ingredient376)
  (clean shaker359)
  (clean shot2)
  (clean shot487)
  (clean shot253)
  (empty shaker359)
  (empty shot2)
  (empty shot487)
  (empty shot253)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail333 ingredient489)
  (cocktail-part2 cocktail333 ingredient348)
)
 (:goal
  (and
      (contains shot2 cocktail333)
      (contains shot487 cocktail333)
)))
