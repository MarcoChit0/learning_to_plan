(define (problem prob)
 (:domain barman)
 (:objects 
      shaker389 - shaker
      left right - hand
      shot26 shot346 - shot
      ingredient67 ingredient339 ingredient78 - ingredient
      cocktail497 - cocktail
      dispenser406 dispenser288 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker389)
  (ontable shot26)
  (ontable shot346)
  (dispenses dispenser406 ingredient67)
  (dispenses dispenser288 ingredient339)
  (dispenses dispenser143 ingredient78)
  (clean shaker389)
  (clean shot26)
  (clean shot346)
  (empty shaker389)
  (empty shot26)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker389 l0)
  (shaker-level shaker389 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail497 ingredient67)
  (cocktail-part2 cocktail497 ingredient339)
)
 (:goal
  (and
      (contains shot26 cocktail497)
)))
