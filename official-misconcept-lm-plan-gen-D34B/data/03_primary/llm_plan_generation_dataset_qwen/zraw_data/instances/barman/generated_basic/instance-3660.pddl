(define (problem prob)
 (:domain barman)
 (:objects 
      shaker289 - shaker
      left right - hand
      shot200 shot27 - shot
      ingredient286 ingredient361 ingredient473 ingredient166 - ingredient
      cocktail414 - cocktail
      dispenser341 dispenser344 dispenser69 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker289)
  (ontable shot200)
  (ontable shot27)
  (dispenses dispenser341 ingredient286)
  (dispenses dispenser344 ingredient361)
  (dispenses dispenser69 ingredient473)
  (dispenses dispenser5 ingredient166)
  (clean shaker289)
  (clean shot200)
  (clean shot27)
  (empty shaker289)
  (empty shot200)
  (empty shot27)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker289 l0)
  (shaker-level shaker289 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail414 ingredient361)
  (cocktail-part2 cocktail414 ingredient166)
)
 (:goal
  (and
      (contains shot200 cocktail414)
)))
