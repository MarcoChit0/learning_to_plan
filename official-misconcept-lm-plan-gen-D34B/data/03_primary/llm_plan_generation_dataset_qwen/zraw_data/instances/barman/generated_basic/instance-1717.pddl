(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot140 shot462 - shot
      ingredient33 ingredient487 ingredient45 ingredient189 - ingredient
      cocktail141 - cocktail
      dispenser371 dispenser346 dispenser322 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot140)
  (ontable shot462)
  (dispenses dispenser371 ingredient33)
  (dispenses dispenser346 ingredient487)
  (dispenses dispenser322 ingredient45)
  (dispenses dispenser283 ingredient189)
  (clean shaker41)
  (clean shot140)
  (clean shot462)
  (empty shaker41)
  (empty shot140)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient189)
  (cocktail-part2 cocktail141 ingredient33)
)
 (:goal
  (and
      (contains shot140 cocktail141)
)))
