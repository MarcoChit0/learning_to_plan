(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot58 shot355 shot369 - shot
      ingredient383 ingredient35 ingredient271 ingredient292 - ingredient
      cocktail1 - cocktail
      dispenser357 dispenser169 dispenser137 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot58)
  (ontable shot355)
  (ontable shot369)
  (dispenses dispenser357 ingredient383)
  (dispenses dispenser169 ingredient35)
  (dispenses dispenser137 ingredient271)
  (dispenses dispenser430 ingredient292)
  (clean shaker55)
  (clean shot58)
  (clean shot355)
  (clean shot369)
  (empty shaker55)
  (empty shot58)
  (empty shot355)
  (empty shot369)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient35)
  (cocktail-part2 cocktail1 ingredient383)
)
 (:goal
  (and
      (contains shot58 cocktail1)
      (contains shot355 ingredient35)
)))
