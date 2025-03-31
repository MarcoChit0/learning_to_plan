(define (problem prob)
 (:domain barman)
 (:objects 
      shaker169 - shaker
      left right - hand
      shot391 shot168 - shot
      ingredient198 ingredient315 - ingredient
      cocktail350 - cocktail
      dispenser285 dispenser265 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker169)
  (ontable shot391)
  (ontable shot168)
  (dispenses dispenser285 ingredient198)
  (dispenses dispenser265 ingredient315)
  (clean shaker169)
  (clean shot391)
  (clean shot168)
  (empty shaker169)
  (empty shot391)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker169 l0)
  (shaker-level shaker169 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient315)
  (cocktail-part2 cocktail350 ingredient198)
)
 (:goal
  (and
      (contains shot391 cocktail350)
)))
