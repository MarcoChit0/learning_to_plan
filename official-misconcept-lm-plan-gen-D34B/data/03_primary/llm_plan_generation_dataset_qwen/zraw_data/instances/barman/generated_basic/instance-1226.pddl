(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot353 shot225 shot311 - shot
      ingredient375 ingredient422 ingredient205 - ingredient
      cocktail208 - cocktail
      dispenser117 dispenser273 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot353)
  (ontable shot225)
  (ontable shot311)
  (dispenses dispenser117 ingredient375)
  (dispenses dispenser273 ingredient422)
  (dispenses dispenser352 ingredient205)
  (clean shaker221)
  (clean shot353)
  (clean shot225)
  (clean shot311)
  (empty shaker221)
  (empty shot353)
  (empty shot225)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail208 ingredient205)
  (cocktail-part2 cocktail208 ingredient375)
)
 (:goal
  (and
      (contains shot353 cocktail208)
      (contains shot225 cocktail208)
)))
