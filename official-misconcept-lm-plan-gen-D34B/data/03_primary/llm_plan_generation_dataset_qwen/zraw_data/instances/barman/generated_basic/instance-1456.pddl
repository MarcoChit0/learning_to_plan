(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot20 shot67 shot319 - shot
      ingredient110 ingredient336 - ingredient
      cocktail222 - cocktail
      dispenser215 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot20)
  (ontable shot67)
  (ontable shot319)
  (dispenses dispenser215 ingredient110)
  (dispenses dispenser21 ingredient336)
  (clean shaker310)
  (clean shot20)
  (clean shot67)
  (clean shot319)
  (empty shaker310)
  (empty shot20)
  (empty shot67)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient336)
  (cocktail-part2 cocktail222 ingredient110)
)
 (:goal
  (and
      (contains shot20 cocktail222)
      (contains shot67 cocktail222)
)))
