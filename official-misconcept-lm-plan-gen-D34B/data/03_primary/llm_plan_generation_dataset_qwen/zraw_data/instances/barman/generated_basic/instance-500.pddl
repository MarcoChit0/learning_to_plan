(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot337 shot207 - shot
      ingredient16 ingredient114 ingredient181 ingredient130 - ingredient
      cocktail1 - cocktail
      dispenser365 dispenser254 dispenser141 dispenser477 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot337)
  (ontable shot207)
  (dispenses dispenser365 ingredient16)
  (dispenses dispenser254 ingredient114)
  (dispenses dispenser141 ingredient181)
  (dispenses dispenser477 ingredient130)
  (clean shaker442)
  (clean shot337)
  (clean shot207)
  (empty shaker442)
  (empty shot337)
  (empty shot207)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient114)
  (cocktail-part2 cocktail1 ingredient16)
)
 (:goal
  (and
      (contains shot337 cocktail1)
)))
