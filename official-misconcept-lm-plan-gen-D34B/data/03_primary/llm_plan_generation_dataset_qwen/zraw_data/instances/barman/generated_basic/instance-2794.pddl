(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot116 - shot
      ingredient40 ingredient273 ingredient216 ingredient188 - ingredient
      cocktail283 - cocktail
      dispenser133 dispenser142 dispenser217 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot116)
  (dispenses dispenser133 ingredient40)
  (dispenses dispenser142 ingredient273)
  (dispenses dispenser217 ingredient216)
  (dispenses dispenser299 ingredient188)
  (clean shaker440)
  (clean shot116)
  (empty shaker440)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail283 ingredient188)
  (cocktail-part2 cocktail283 ingredient40)
)
 (:goal
  (and
      (contains shot116 cocktail283)
)))
