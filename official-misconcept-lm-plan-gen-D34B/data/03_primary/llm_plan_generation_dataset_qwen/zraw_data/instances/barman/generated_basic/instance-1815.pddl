(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot291 shot324 shot229 - shot
      ingredient430 ingredient355 ingredient414 ingredient210 - ingredient
      cocktail96 - cocktail
      dispenser451 dispenser40 dispenser101 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot291)
  (ontable shot324)
  (ontable shot229)
  (dispenses dispenser451 ingredient430)
  (dispenses dispenser40 ingredient355)
  (dispenses dispenser101 ingredient414)
  (dispenses dispenser398 ingredient210)
  (clean shaker25)
  (clean shot291)
  (clean shot324)
  (clean shot229)
  (empty shaker25)
  (empty shot291)
  (empty shot324)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail96 ingredient210)
  (cocktail-part2 cocktail96 ingredient430)
)
 (:goal
  (and
      (contains shot291 cocktail96)
      (contains shot324 cocktail96)
)))
