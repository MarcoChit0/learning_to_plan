(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot139 shot309 shot431 - shot
      ingredient390 ingredient177 - ingredient
      cocktail279 - cocktail
      dispenser148 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot139)
  (ontable shot309)
  (ontable shot431)
  (dispenses dispenser148 ingredient390)
  (dispenses dispenser132 ingredient177)
  (clean shaker385)
  (clean shot139)
  (clean shot309)
  (clean shot431)
  (empty shaker385)
  (empty shot139)
  (empty shot309)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail279 ingredient177)
  (cocktail-part2 cocktail279 ingredient390)
)
 (:goal
  (and
      (contains shot139 cocktail279)
      (contains shot309 cocktail279)
)))
