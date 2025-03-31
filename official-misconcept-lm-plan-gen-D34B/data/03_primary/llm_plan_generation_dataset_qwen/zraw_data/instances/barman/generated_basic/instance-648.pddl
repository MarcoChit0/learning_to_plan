(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot95 shot117 - shot
      ingredient230 ingredient366 ingredient256 - ingredient
      cocktail1 - cocktail
      dispenser329 dispenser7 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot95)
  (ontable shot117)
  (dispenses dispenser329 ingredient230)
  (dispenses dispenser7 ingredient366)
  (dispenses dispenser221 ingredient256)
  (clean shaker143)
  (clean shot95)
  (clean shot117)
  (empty shaker143)
  (empty shot95)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient256)
  (cocktail-part2 cocktail1 ingredient230)
)
 (:goal
  (and
      (contains shot95 cocktail1)
)))
