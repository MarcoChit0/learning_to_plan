(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot345 shot450 - shot
      ingredient377 ingredient153 ingredient35 ingredient105 - ingredient
      cocktail83 - cocktail
      dispenser97 dispenser275 dispenser233 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot345)
  (ontable shot450)
  (dispenses dispenser97 ingredient377)
  (dispenses dispenser275 ingredient153)
  (dispenses dispenser233 ingredient35)
  (dispenses dispenser420 ingredient105)
  (clean shaker377)
  (clean shot345)
  (clean shot450)
  (empty shaker377)
  (empty shot345)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient105)
  (cocktail-part2 cocktail83 ingredient153)
)
 (:goal
  (and
      (contains shot345 cocktail83)
)))
