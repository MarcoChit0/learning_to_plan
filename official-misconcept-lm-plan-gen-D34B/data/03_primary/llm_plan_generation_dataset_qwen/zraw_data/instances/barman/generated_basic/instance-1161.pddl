(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot323 shot306 - shot
      ingredient69 ingredient497 ingredient146 ingredient105 - ingredient
      cocktail1 - cocktail
      dispenser401 dispenser184 dispenser113 dispenser78 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot323)
  (ontable shot306)
  (dispenses dispenser401 ingredient69)
  (dispenses dispenser184 ingredient497)
  (dispenses dispenser113 ingredient146)
  (dispenses dispenser78 ingredient105)
  (clean shaker132)
  (clean shot323)
  (clean shot306)
  (empty shaker132)
  (empty shot323)
  (empty shot306)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient105)
)
 (:goal
  (and
      (contains shot323 cocktail1)
)))
