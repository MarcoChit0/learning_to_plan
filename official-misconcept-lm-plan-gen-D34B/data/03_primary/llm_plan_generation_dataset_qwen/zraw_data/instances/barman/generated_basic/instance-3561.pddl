(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot240 shot175 - shot
      ingredient95 ingredient109 ingredient413 ingredient52 - ingredient
      cocktail87 - cocktail
      dispenser98 dispenser367 dispenser69 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot240)
  (ontable shot175)
  (dispenses dispenser98 ingredient95)
  (dispenses dispenser367 ingredient109)
  (dispenses dispenser69 ingredient413)
  (dispenses dispenser436 ingredient52)
  (clean shaker61)
  (clean shot240)
  (clean shot175)
  (empty shaker61)
  (empty shot240)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient109)
  (cocktail-part2 cocktail87 ingredient52)
)
 (:goal
  (and
      (contains shot240 cocktail87)
)))
