(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot217 shot417 - shot
      ingredient395 ingredient73 - ingredient
      cocktail460 - cocktail
      dispenser33 dispenser254 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot217)
  (ontable shot417)
  (dispenses dispenser33 ingredient395)
  (dispenses dispenser254 ingredient73)
  (clean shaker115)
  (clean shot217)
  (clean shot417)
  (empty shaker115)
  (empty shot217)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient73)
  (cocktail-part2 cocktail460 ingredient395)
)
 (:goal
  (and
      (contains shot217 cocktail460)
)))
