(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot138 - shot
      ingredient112 ingredient268 ingredient487 ingredient362 - ingredient
      cocktail277 - cocktail
      dispenser144 dispenser497 dispenser178 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot138)
  (dispenses dispenser144 ingredient112)
  (dispenses dispenser497 ingredient268)
  (dispenses dispenser178 ingredient487)
  (dispenses dispenser23 ingredient362)
  (clean shaker332)
  (clean shot138)
  (empty shaker332)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient362)
  (cocktail-part2 cocktail277 ingredient268)
)
 (:goal
  (and
      (contains shot138 cocktail277)
)))
