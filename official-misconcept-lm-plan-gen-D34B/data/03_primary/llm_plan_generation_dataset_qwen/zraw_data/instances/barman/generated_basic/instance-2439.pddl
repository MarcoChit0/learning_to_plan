(define (problem prob)
 (:domain barman)
 (:objects 
      shaker463 - shaker
      left right - hand
      shot291 shot413 shot132 - shot
      ingredient161 ingredient271 ingredient330 - ingredient
      cocktail140 - cocktail
      dispenser19 dispenser227 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker463)
  (ontable shot291)
  (ontable shot413)
  (ontable shot132)
  (dispenses dispenser19 ingredient161)
  (dispenses dispenser227 ingredient271)
  (dispenses dispenser155 ingredient330)
  (clean shaker463)
  (clean shot291)
  (clean shot413)
  (clean shot132)
  (empty shaker463)
  (empty shot291)
  (empty shot413)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker463 l0)
  (shaker-level shaker463 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail140 ingredient161)
  (cocktail-part2 cocktail140 ingredient271)
)
 (:goal
  (and
      (contains shot291 cocktail140)
      (contains shot413 cocktail140)
)))
