(define (problem prob)
 (:domain barman)
 (:objects 
      shaker11 - shaker
      left right - hand
      shot213 shot177 - shot
      ingredient198 ingredient358 - ingredient
      cocktail397 - cocktail
      dispenser272 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker11)
  (ontable shot213)
  (ontable shot177)
  (dispenses dispenser272 ingredient198)
  (dispenses dispenser341 ingredient358)
  (clean shaker11)
  (clean shot213)
  (clean shot177)
  (empty shaker11)
  (empty shot213)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker11 l0)
  (shaker-level shaker11 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient198)
  (cocktail-part2 cocktail397 ingredient358)
)
 (:goal
  (and
      (contains shot213 cocktail397)
)))
