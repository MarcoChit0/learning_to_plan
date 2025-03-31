(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot238 shot160 shot52 - shot
      ingredient140 ingredient101 ingredient296 - ingredient
      cocktail346 - cocktail
      dispenser109 dispenser150 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot238)
  (ontable shot160)
  (ontable shot52)
  (dispenses dispenser109 ingredient140)
  (dispenses dispenser150 ingredient101)
  (dispenses dispenser15 ingredient296)
  (clean shaker94)
  (clean shot238)
  (clean shot160)
  (clean shot52)
  (empty shaker94)
  (empty shot238)
  (empty shot160)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail346 ingredient140)
  (cocktail-part2 cocktail346 ingredient296)
)
 (:goal
  (and
      (contains shot238 cocktail346)
      (contains shot160 cocktail346)
)))
