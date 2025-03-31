(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot26 shot480 - shot
      ingredient76 ingredient478 ingredient215 ingredient414 - ingredient
      cocktail210 - cocktail
      dispenser414 dispenser254 dispenser386 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot26)
  (ontable shot480)
  (dispenses dispenser414 ingredient76)
  (dispenses dispenser254 ingredient478)
  (dispenses dispenser386 ingredient215)
  (dispenses dispenser458 ingredient414)
  (clean shaker179)
  (clean shot26)
  (clean shot480)
  (empty shaker179)
  (empty shot26)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient215)
  (cocktail-part2 cocktail210 ingredient414)
)
 (:goal
  (and
      (contains shot26 cocktail210)
)))
