(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot337 shot315 shot392 - shot
      ingredient9 ingredient96 ingredient243 ingredient431 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser356 dispenser124 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot337)
  (ontable shot315)
  (ontable shot392)
  (dispenses dispenser366 ingredient9)
  (dispenses dispenser356 ingredient96)
  (dispenses dispenser124 ingredient243)
  (dispenses dispenser209 ingredient431)
  (clean shaker163)
  (clean shot337)
  (clean shot315)
  (clean shot392)
  (empty shaker163)
  (empty shot337)
  (empty shot315)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient9)
  (cocktail-part2 cocktail1 ingredient431)
)
 (:goal
  (and
      (contains shot337 cocktail1)
      (contains shot315 cocktail1)
)))
