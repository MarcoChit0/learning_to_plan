(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot103 shot466 shot317 - shot
      ingredient183 ingredient324 ingredient273 ingredient375 - ingredient
      cocktail423 - cocktail
      dispenser449 dispenser232 dispenser169 dispenser12 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot103)
  (ontable shot466)
  (ontable shot317)
  (dispenses dispenser449 ingredient183)
  (dispenses dispenser232 ingredient324)
  (dispenses dispenser169 ingredient273)
  (dispenses dispenser12 ingredient375)
  (clean shaker0)
  (clean shot103)
  (clean shot466)
  (clean shot317)
  (empty shaker0)
  (empty shot103)
  (empty shot466)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient375)
  (cocktail-part2 cocktail423 ingredient273)
)
 (:goal
  (and
      (contains shot103 cocktail423)
      (contains shot466 cocktail423)
)))
