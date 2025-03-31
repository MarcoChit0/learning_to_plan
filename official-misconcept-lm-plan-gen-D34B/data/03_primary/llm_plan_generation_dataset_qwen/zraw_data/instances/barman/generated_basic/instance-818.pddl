(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot319 shot487 - shot
      ingredient209 ingredient243 ingredient296 ingredient44 - ingredient
      cocktail1 - cocktail
      dispenser249 dispenser372 dispenser215 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot319)
  (ontable shot487)
  (dispenses dispenser249 ingredient209)
  (dispenses dispenser372 ingredient243)
  (dispenses dispenser215 ingredient296)
  (dispenses dispenser431 ingredient44)
  (clean shaker270)
  (clean shot319)
  (clean shot487)
  (empty shaker270)
  (empty shot319)
  (empty shot487)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient44)
  (cocktail-part2 cocktail1 ingredient209)
)
 (:goal
  (and
      (contains shot319 cocktail1)
)))
