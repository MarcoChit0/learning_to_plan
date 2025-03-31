(define (problem prob)
 (:domain barman)
 (:objects 
      shaker145 - shaker
      left right - hand
      shot115 shot147 - shot
      ingredient128 ingredient361 - ingredient
      cocktail41 - cocktail
      dispenser415 dispenser253 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker145)
  (ontable shot115)
  (ontable shot147)
  (dispenses dispenser415 ingredient128)
  (dispenses dispenser253 ingredient361)
  (clean shaker145)
  (clean shot115)
  (clean shot147)
  (empty shaker145)
  (empty shot115)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker145 l0)
  (shaker-level shaker145 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient128)
  (cocktail-part2 cocktail41 ingredient361)
)
 (:goal
  (and
      (contains shot115 cocktail41)
)))
