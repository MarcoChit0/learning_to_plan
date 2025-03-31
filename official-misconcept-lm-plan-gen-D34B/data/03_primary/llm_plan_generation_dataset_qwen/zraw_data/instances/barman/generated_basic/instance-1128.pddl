(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot472 shot400 - shot
      ingredient9 ingredient230 ingredient106 - ingredient
      cocktail1 - cocktail
      dispenser263 dispenser367 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot472)
  (ontable shot400)
  (dispenses dispenser263 ingredient9)
  (dispenses dispenser367 ingredient230)
  (dispenses dispenser223 ingredient106)
  (clean shaker228)
  (clean shot472)
  (clean shot400)
  (empty shaker228)
  (empty shot472)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient230)
  (cocktail-part2 cocktail1 ingredient106)
)
 (:goal
  (and
      (contains shot472 cocktail1)
)))
