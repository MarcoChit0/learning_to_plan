(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot278 shot396 shot289 - shot
      ingredient240 ingredient191 ingredient261 - ingredient
      cocktail263 - cocktail
      dispenser290 dispenser490 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot278)
  (ontable shot396)
  (ontable shot289)
  (dispenses dispenser290 ingredient240)
  (dispenses dispenser490 ingredient191)
  (dispenses dispenser340 ingredient261)
  (clean shaker359)
  (clean shot278)
  (clean shot396)
  (clean shot289)
  (empty shaker359)
  (empty shot278)
  (empty shot396)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail263 ingredient240)
  (cocktail-part2 cocktail263 ingredient191)
)
 (:goal
  (and
      (contains shot278 cocktail263)
      (contains shot396 cocktail263)
)))
