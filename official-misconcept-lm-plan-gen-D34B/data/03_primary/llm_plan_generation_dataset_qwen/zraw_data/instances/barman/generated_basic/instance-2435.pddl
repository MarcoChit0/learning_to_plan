(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot417 shot78 shot230 - shot
      ingredient218 ingredient146 ingredient94 - ingredient
      cocktail425 - cocktail
      dispenser126 dispenser401 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot417)
  (ontable shot78)
  (ontable shot230)
  (dispenses dispenser126 ingredient218)
  (dispenses dispenser401 ingredient146)
  (dispenses dispenser63 ingredient94)
  (clean shaker397)
  (clean shot417)
  (clean shot78)
  (clean shot230)
  (empty shaker397)
  (empty shot417)
  (empty shot78)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail425 ingredient218)
  (cocktail-part2 cocktail425 ingredient146)
)
 (:goal
  (and
      (contains shot417 cocktail425)
      (contains shot78 cocktail425)
)))
