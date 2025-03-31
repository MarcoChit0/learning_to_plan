(define (problem prob)
 (:domain barman)
 (:objects 
      shaker127 - shaker
      left right - hand
      shot64 - shot
      ingredient461 ingredient271 ingredient222 - ingredient
      cocktail38 - cocktail
      dispenser187 dispenser266 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker127)
  (ontable shot64)
  (dispenses dispenser187 ingredient461)
  (dispenses dispenser266 ingredient271)
  (dispenses dispenser245 ingredient222)
  (clean shaker127)
  (clean shot64)
  (empty shaker127)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker127 l0)
  (shaker-level shaker127 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail38 ingredient271)
  (cocktail-part2 cocktail38 ingredient461)
)
 (:goal
  (and
      (contains shot64 cocktail38)
)))
