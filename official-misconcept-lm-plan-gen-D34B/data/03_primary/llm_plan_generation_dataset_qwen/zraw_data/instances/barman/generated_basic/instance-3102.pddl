(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot456 - shot
      ingredient384 ingredient75 ingredient352 ingredient42 - ingredient
      cocktail388 - cocktail
      dispenser5 dispenser25 dispenser164 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot456)
  (dispenses dispenser5 ingredient384)
  (dispenses dispenser25 ingredient75)
  (dispenses dispenser164 ingredient352)
  (dispenses dispenser397 ingredient42)
  (clean shaker425)
  (clean shot456)
  (empty shaker425)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient75)
  (cocktail-part2 cocktail388 ingredient384)
)
 (:goal
  (and
      (contains shot456 cocktail388)
)))
