(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot63 shot2 - shot
      ingredient479 ingredient441 ingredient82 - ingredient
      cocktail1 - cocktail
      dispenser363 dispenser280 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot63)
  (ontable shot2)
  (dispenses dispenser363 ingredient479)
  (dispenses dispenser280 ingredient441)
  (dispenses dispenser206 ingredient82)
  (clean shaker110)
  (clean shot63)
  (clean shot2)
  (empty shaker110)
  (empty shot63)
  (empty shot2)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient82)
  (cocktail-part2 cocktail1 ingredient441)
)
 (:goal
  (and
      (contains shot63 cocktail1)
)))
