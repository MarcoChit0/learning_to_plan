(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot286 shot208 - shot
      ingredient407 ingredient23 ingredient250 - ingredient
      cocktail267 - cocktail
      dispenser23 dispenser233 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot286)
  (ontable shot208)
  (dispenses dispenser23 ingredient407)
  (dispenses dispenser233 ingredient23)
  (dispenses dispenser139 ingredient250)
  (clean shaker433)
  (clean shot286)
  (clean shot208)
  (empty shaker433)
  (empty shot286)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient407)
  (cocktail-part2 cocktail267 ingredient23)
)
 (:goal
  (and
      (contains shot286 cocktail267)
)))
