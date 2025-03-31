(define (problem prob)
 (:domain barman)
 (:objects 
      shaker267 - shaker
      left right - hand
      shot282 shot61 - shot
      ingredient260 ingredient365 - ingredient
      cocktail430 - cocktail
      dispenser460 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker267)
  (ontable shot282)
  (ontable shot61)
  (dispenses dispenser460 ingredient260)
  (dispenses dispenser305 ingredient365)
  (clean shaker267)
  (clean shot282)
  (clean shot61)
  (empty shaker267)
  (empty shot282)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker267 l0)
  (shaker-level shaker267 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient260)
  (cocktail-part2 cocktail430 ingredient365)
)
 (:goal
  (and
      (contains shot282 cocktail430)
)))
