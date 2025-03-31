(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot121 shot497 shot260 - shot
      ingredient213 ingredient32 ingredient379 - ingredient
      cocktail38 - cocktail
      dispenser88 dispenser349 dispenser491 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot121)
  (ontable shot497)
  (ontable shot260)
  (dispenses dispenser88 ingredient213)
  (dispenses dispenser349 ingredient32)
  (dispenses dispenser491 ingredient379)
  (clean shaker112)
  (clean shot121)
  (clean shot497)
  (clean shot260)
  (empty shaker112)
  (empty shot121)
  (empty shot497)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail38 ingredient32)
  (cocktail-part2 cocktail38 ingredient213)
)
 (:goal
  (and
      (contains shot121 cocktail38)
      (contains shot497 cocktail38)
)))
