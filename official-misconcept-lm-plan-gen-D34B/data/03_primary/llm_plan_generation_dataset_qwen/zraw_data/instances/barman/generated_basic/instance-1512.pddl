(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot246 shot87 - shot
      ingredient88 ingredient80 ingredient444 ingredient145 - ingredient
      cocktail389 - cocktail
      dispenser88 dispenser281 dispenser165 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot246)
  (ontable shot87)
  (dispenses dispenser88 ingredient88)
  (dispenses dispenser281 ingredient80)
  (dispenses dispenser165 ingredient444)
  (dispenses dispenser213 ingredient145)
  (clean shaker177)
  (clean shot246)
  (clean shot87)
  (empty shaker177)
  (empty shot246)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail389 ingredient88)
  (cocktail-part2 cocktail389 ingredient145)
)
 (:goal
  (and
      (contains shot246 cocktail389)
)))
