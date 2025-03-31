(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot192 shot401 shot8 - shot
      ingredient415 ingredient487 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser392 dispenser10 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot192)
  (ontable shot401)
  (ontable shot8)
  (dispenses dispenser392 ingredient415)
  (dispenses dispenser10 ingredient487)
  (clean shaker364)
  (clean shot192)
  (clean shot401)
  (clean shot8)
  (empty shaker364)
  (empty shot192)
  (empty shot401)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient415)
  (cocktail-part2 cocktail1 ingredient487)
  (cocktail-part1 cocktail2 ingredient487)
  (cocktail-part2 cocktail2 ingredient415)
)
 (:goal
  (and
      (contains shot192 cocktail1)
      (contains shot401 cocktail2)
)))
