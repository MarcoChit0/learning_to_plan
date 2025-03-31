(define (problem prob)
 (:domain barman)
 (:objects 
      shaker359 - shaker
      left right - hand
      shot97 shot389 shot359 - shot
      ingredient184 ingredient212 - ingredient
      cocktail292 - cocktail
      dispenser227 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker359)
  (ontable shot97)
  (ontable shot389)
  (ontable shot359)
  (dispenses dispenser227 ingredient184)
  (dispenses dispenser100 ingredient212)
  (clean shaker359)
  (clean shot97)
  (clean shot389)
  (clean shot359)
  (empty shaker359)
  (empty shot97)
  (empty shot389)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker359 l0)
  (shaker-level shaker359 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient212)
  (cocktail-part2 cocktail292 ingredient184)
)
 (:goal
  (and
      (contains shot97 cocktail292)
      (contains shot389 cocktail292)
)))
