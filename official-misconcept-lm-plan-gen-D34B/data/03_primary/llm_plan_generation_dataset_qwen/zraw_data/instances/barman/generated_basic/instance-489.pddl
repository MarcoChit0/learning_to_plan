(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot247 shot340 - shot
      ingredient430 ingredient107 ingredient220 ingredient278 - ingredient
      cocktail1 - cocktail
      dispenser143 dispenser88 dispenser405 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot247)
  (ontable shot340)
  (dispenses dispenser143 ingredient430)
  (dispenses dispenser88 ingredient107)
  (dispenses dispenser405 ingredient220)
  (dispenses dispenser232 ingredient278)
  (clean shaker383)
  (clean shot247)
  (clean shot340)
  (empty shaker383)
  (empty shot247)
  (empty shot340)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient278)
  (cocktail-part2 cocktail1 ingredient430)
)
 (:goal
  (and
      (contains shot247 cocktail1)
)))
