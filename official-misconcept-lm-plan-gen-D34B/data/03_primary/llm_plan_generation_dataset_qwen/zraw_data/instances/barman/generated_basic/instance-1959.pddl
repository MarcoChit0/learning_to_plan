(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot332 shot313 shot389 - shot
      ingredient299 ingredient38 ingredient40 ingredient295 - ingredient
      cocktail481 - cocktail
      dispenser444 dispenser44 dispenser56 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot332)
  (ontable shot313)
  (ontable shot389)
  (dispenses dispenser444 ingredient299)
  (dispenses dispenser44 ingredient38)
  (dispenses dispenser56 ingredient40)
  (dispenses dispenser103 ingredient295)
  (clean shaker67)
  (clean shot332)
  (clean shot313)
  (clean shot389)
  (empty shaker67)
  (empty shot332)
  (empty shot313)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail481 ingredient299)
  (cocktail-part2 cocktail481 ingredient38)
)
 (:goal
  (and
      (contains shot332 cocktail481)
      (contains shot313 cocktail481)
)))
