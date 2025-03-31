(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot78 shot175 shot242 - shot
      ingredient120 ingredient33 - ingredient
      cocktail184 - cocktail
      dispenser59 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot78)
  (ontable shot175)
  (ontable shot242)
  (dispenses dispenser59 ingredient120)
  (dispenses dispenser5 ingredient33)
  (clean shaker57)
  (clean shot78)
  (clean shot175)
  (clean shot242)
  (empty shaker57)
  (empty shot78)
  (empty shot175)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient120)
  (cocktail-part2 cocktail184 ingredient33)
)
 (:goal
  (and
      (contains shot78 cocktail184)
      (contains shot175 cocktail184)
)))
