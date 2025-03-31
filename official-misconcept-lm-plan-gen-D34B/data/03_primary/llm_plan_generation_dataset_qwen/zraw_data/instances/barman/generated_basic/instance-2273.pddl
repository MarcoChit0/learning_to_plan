(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot106 shot355 - shot
      ingredient34 ingredient7 ingredient21 ingredient350 - ingredient
      cocktail139 - cocktail
      dispenser367 dispenser309 dispenser31 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot106)
  (ontable shot355)
  (dispenses dispenser367 ingredient34)
  (dispenses dispenser309 ingredient7)
  (dispenses dispenser31 ingredient21)
  (dispenses dispenser269 ingredient350)
  (clean shaker398)
  (clean shot106)
  (clean shot355)
  (empty shaker398)
  (empty shot106)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail139 ingredient34)
  (cocktail-part2 cocktail139 ingredient7)
)
 (:goal
  (and
      (contains shot106 cocktail139)
)))
