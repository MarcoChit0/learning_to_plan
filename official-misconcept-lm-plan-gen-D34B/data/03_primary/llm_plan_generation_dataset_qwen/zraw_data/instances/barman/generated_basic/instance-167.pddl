(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot272 shot57 - shot
      ingredient424 ingredient399 ingredient69 - ingredient
      cocktail1 - cocktail
      dispenser127 dispenser324 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot272)
  (ontable shot57)
  (dispenses dispenser127 ingredient424)
  (dispenses dispenser324 ingredient399)
  (dispenses dispenser340 ingredient69)
  (clean shaker149)
  (clean shot272)
  (clean shot57)
  (empty shaker149)
  (empty shot272)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient69)
  (cocktail-part2 cocktail1 ingredient424)
)
 (:goal
  (and
      (contains shot272 cocktail1)
)))
