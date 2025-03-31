(define (problem prob)
 (:domain barman)
 (:objects 
      shaker245 - shaker
      left right - hand
      shot61 shot107 shot373 - shot
      ingredient462 ingredient154 ingredient4 - ingredient
      cocktail193 - cocktail
      dispenser17 dispenser71 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker245)
  (ontable shot61)
  (ontable shot107)
  (ontable shot373)
  (dispenses dispenser17 ingredient462)
  (dispenses dispenser71 ingredient154)
  (dispenses dispenser442 ingredient4)
  (clean shaker245)
  (clean shot61)
  (clean shot107)
  (clean shot373)
  (empty shaker245)
  (empty shot61)
  (empty shot107)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker245 l0)
  (shaker-level shaker245 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail193 ingredient4)
  (cocktail-part2 cocktail193 ingredient462)
)
 (:goal
  (and
      (contains shot61 cocktail193)
      (contains shot107 cocktail193)
)))
