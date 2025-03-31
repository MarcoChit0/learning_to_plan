(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot143 - shot
      ingredient231 ingredient449 ingredient283 ingredient396 - ingredient
      cocktail296 - cocktail
      dispenser142 dispenser489 dispenser179 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot143)
  (dispenses dispenser142 ingredient231)
  (dispenses dispenser489 ingredient449)
  (dispenses dispenser179 ingredient283)
  (dispenses dispenser115 ingredient396)
  (clean shaker411)
  (clean shot143)
  (empty shaker411)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail296 ingredient283)
  (cocktail-part2 cocktail296 ingredient396)
)
 (:goal
  (and
      (contains shot143 cocktail296)
)))
