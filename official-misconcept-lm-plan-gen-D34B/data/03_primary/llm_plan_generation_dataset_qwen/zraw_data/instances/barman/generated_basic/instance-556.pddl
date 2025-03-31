(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot455 shot57 - shot
      ingredient470 ingredient212 ingredient244 - ingredient
      cocktail1 - cocktail
      dispenser231 dispenser146 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot455)
  (ontable shot57)
  (dispenses dispenser231 ingredient470)
  (dispenses dispenser146 ingredient212)
  (dispenses dispenser363 ingredient244)
  (clean shaker72)
  (clean shot455)
  (clean shot57)
  (empty shaker72)
  (empty shot455)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient244)
  (cocktail-part2 cocktail1 ingredient470)
)
 (:goal
  (and
      (contains shot455 cocktail1)
)))
