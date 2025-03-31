(define (problem prob)
 (:domain barman)
 (:objects 
      shaker33 - shaker
      left right - hand
      shot303 shot130 shot345 - shot
      ingredient354 ingredient221 ingredient76 - ingredient
      cocktail270 - cocktail
      dispenser172 dispenser386 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker33)
  (ontable shot303)
  (ontable shot130)
  (ontable shot345)
  (dispenses dispenser172 ingredient354)
  (dispenses dispenser386 ingredient221)
  (dispenses dispenser484 ingredient76)
  (clean shaker33)
  (clean shot303)
  (clean shot130)
  (clean shot345)
  (empty shaker33)
  (empty shot303)
  (empty shot130)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker33 l0)
  (shaker-level shaker33 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail270 ingredient76)
  (cocktail-part2 cocktail270 ingredient354)
)
 (:goal
  (and
      (contains shot303 cocktail270)
      (contains shot130 ingredient76)
)))
