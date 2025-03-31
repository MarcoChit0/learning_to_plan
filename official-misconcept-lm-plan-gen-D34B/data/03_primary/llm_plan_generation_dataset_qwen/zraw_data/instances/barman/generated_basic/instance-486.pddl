(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot375 shot337 - shot
      ingredient149 ingredient407 ingredient241 ingredient83 - ingredient
      cocktail1 - cocktail
      dispenser468 dispenser486 dispenser433 dispenser199 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot375)
  (ontable shot337)
  (dispenses dispenser468 ingredient149)
  (dispenses dispenser486 ingredient407)
  (dispenses dispenser433 ingredient241)
  (dispenses dispenser199 ingredient83)
  (clean shaker273)
  (clean shot375)
  (clean shot337)
  (empty shaker273)
  (empty shot375)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient83)
  (cocktail-part2 cocktail1 ingredient149)
)
 (:goal
  (and
      (contains shot375 cocktail1)
)))
