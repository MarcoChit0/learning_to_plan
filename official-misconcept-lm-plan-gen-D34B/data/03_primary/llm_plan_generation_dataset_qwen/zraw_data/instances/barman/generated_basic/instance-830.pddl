(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot61 shot242 shot350 - shot
      ingredient208 ingredient59 ingredient143 ingredient454 - ingredient
      cocktail1 - cocktail
      dispenser162 dispenser197 dispenser174 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot61)
  (ontable shot242)
  (ontable shot350)
  (dispenses dispenser162 ingredient208)
  (dispenses dispenser197 ingredient59)
  (dispenses dispenser174 ingredient143)
  (dispenses dispenser191 ingredient454)
  (clean shaker347)
  (clean shot61)
  (clean shot242)
  (clean shot350)
  (empty shaker347)
  (empty shot61)
  (empty shot242)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient143)
  (cocktail-part2 cocktail1 ingredient59)
)
 (:goal
  (and
      (contains shot61 cocktail1)
      (contains shot242 cocktail1)
)))
