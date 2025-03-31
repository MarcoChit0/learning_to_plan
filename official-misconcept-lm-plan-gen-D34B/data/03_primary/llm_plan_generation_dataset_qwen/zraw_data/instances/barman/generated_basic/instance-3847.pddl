(define (problem prob)
 (:domain barman)
 (:objects 
      shaker349 - shaker
      left right - hand
      shot159 shot436 shot321 - shot
      ingredient96 ingredient496 - ingredient
      cocktail223 - cocktail
      dispenser482 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker349)
  (ontable shot159)
  (ontable shot436)
  (ontable shot321)
  (dispenses dispenser482 ingredient96)
  (dispenses dispenser260 ingredient496)
  (clean shaker349)
  (clean shot159)
  (clean shot436)
  (clean shot321)
  (empty shaker349)
  (empty shot159)
  (empty shot436)
  (empty shot321)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker349 l0)
  (shaker-level shaker349 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient96)
  (cocktail-part2 cocktail223 ingredient496)
)
 (:goal
  (and
      (contains shot159 cocktail223)
      (contains shot436 cocktail223)
)))
