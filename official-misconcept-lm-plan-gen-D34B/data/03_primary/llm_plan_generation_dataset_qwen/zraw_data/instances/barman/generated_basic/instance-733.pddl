(define (problem prob)
 (:domain barman)
 (:objects 
      shaker258 - shaker
      left right - hand
      shot192 - shot
      ingredient85 ingredient449 ingredient428 - ingredient
      cocktail1 - cocktail
      dispenser122 dispenser203 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker258)
  (ontable shot192)
  (dispenses dispenser122 ingredient85)
  (dispenses dispenser203 ingredient449)
  (dispenses dispenser342 ingredient428)
  (clean shaker258)
  (clean shot192)
  (empty shaker258)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker258 l0)
  (shaker-level shaker258 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient85)
  (cocktail-part2 cocktail1 ingredient428)
)
 (:goal
  (and
      (contains shot192 cocktail1)
)))
