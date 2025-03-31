(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot424 shot323 - shot
      ingredient114 ingredient321 - ingredient
      cocktail386 - cocktail
      dispenser436 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot424)
  (ontable shot323)
  (dispenses dispenser436 ingredient114)
  (dispenses dispenser405 ingredient321)
  (clean shaker73)
  (clean shot424)
  (clean shot323)
  (empty shaker73)
  (empty shot424)
  (empty shot323)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient114)
  (cocktail-part2 cocktail386 ingredient321)
)
 (:goal
  (and
      (contains shot424 cocktail386)
)))
