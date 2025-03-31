(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot375 shot170 - shot
      ingredient70 ingredient21 ingredient114 - ingredient
      cocktail464 - cocktail
      dispenser371 dispenser146 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot375)
  (ontable shot170)
  (dispenses dispenser371 ingredient70)
  (dispenses dispenser146 ingredient21)
  (dispenses dispenser387 ingredient114)
  (clean shaker398)
  (clean shot375)
  (clean shot170)
  (empty shaker398)
  (empty shot375)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail464 ingredient21)
  (cocktail-part2 cocktail464 ingredient114)
)
 (:goal
  (and
      (contains shot375 cocktail464)
)))
