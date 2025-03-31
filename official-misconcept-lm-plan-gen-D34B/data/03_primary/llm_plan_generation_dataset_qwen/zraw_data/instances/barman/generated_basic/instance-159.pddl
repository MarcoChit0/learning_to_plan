(define (problem prob)
 (:domain barman)
 (:objects 
      shaker100 - shaker
      left right - hand
      shot392 shot461 - shot
      ingredient282 ingredient232 ingredient436 - ingredient
      cocktail1 - cocktail
      dispenser152 dispenser33 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker100)
  (ontable shot392)
  (ontable shot461)
  (dispenses dispenser152 ingredient282)
  (dispenses dispenser33 ingredient232)
  (dispenses dispenser41 ingredient436)
  (clean shaker100)
  (clean shot392)
  (clean shot461)
  (empty shaker100)
  (empty shot392)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker100 l0)
  (shaker-level shaker100 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient232)
  (cocktail-part2 cocktail1 ingredient282)
)
 (:goal
  (and
      (contains shot392 cocktail1)
)))
