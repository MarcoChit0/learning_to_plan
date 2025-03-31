(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot247 shot185 - shot
      ingredient474 ingredient344 - ingredient
      cocktail1 - cocktail
      dispenser177 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot247)
  (ontable shot185)
  (dispenses dispenser177 ingredient474)
  (dispenses dispenser44 ingredient344)
  (clean shaker213)
  (clean shot247)
  (clean shot185)
  (empty shaker213)
  (empty shot247)
  (empty shot185)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient474)
  (cocktail-part2 cocktail1 ingredient344)
)
 (:goal
  (and
      (contains shot247 cocktail1)
)))
