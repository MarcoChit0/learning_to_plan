(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot278 shot36 - shot
      ingredient369 ingredient56 ingredient266 - ingredient
      cocktail277 - cocktail
      dispenser113 dispenser283 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot278)
  (ontable shot36)
  (dispenses dispenser113 ingredient369)
  (dispenses dispenser283 ingredient56)
  (dispenses dispenser130 ingredient266)
  (clean shaker51)
  (clean shot278)
  (clean shot36)
  (empty shaker51)
  (empty shot278)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient369)
  (cocktail-part2 cocktail277 ingredient56)
)
 (:goal
  (and
      (contains shot278 cocktail277)
)))
