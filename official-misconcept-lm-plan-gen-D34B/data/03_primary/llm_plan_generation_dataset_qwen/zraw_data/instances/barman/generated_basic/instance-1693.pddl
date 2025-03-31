(define (problem prob)
 (:domain barman)
 (:objects 
      shaker137 - shaker
      left right - hand
      shot414 shot54 shot319 shot440 - shot
      ingredient377 ingredient477 ingredient340 - ingredient
      cocktail222 - cocktail
      dispenser259 dispenser243 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker137)
  (ontable shot414)
  (ontable shot54)
  (ontable shot319)
  (ontable shot440)
  (dispenses dispenser259 ingredient377)
  (dispenses dispenser243 ingredient477)
  (dispenses dispenser214 ingredient340)
  (clean shaker137)
  (clean shot414)
  (clean shot54)
  (clean shot319)
  (clean shot440)
  (empty shaker137)
  (empty shot414)
  (empty shot54)
  (empty shot319)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker137 l0)
  (shaker-level shaker137 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail222 ingredient377)
  (cocktail-part2 cocktail222 ingredient340)
)
 (:goal
  (and
      (contains shot414 cocktail222)
      (contains shot54 cocktail222)
      (contains shot319 ingredient340)
)))
