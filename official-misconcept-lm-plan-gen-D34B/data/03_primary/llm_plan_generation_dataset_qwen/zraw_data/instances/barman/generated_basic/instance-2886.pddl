(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot402 shot48 shot238 - shot
      ingredient144 ingredient252 - ingredient
      cocktail110 - cocktail
      dispenser464 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot402)
  (ontable shot48)
  (ontable shot238)
  (dispenses dispenser464 ingredient144)
  (dispenses dispenser246 ingredient252)
  (clean shaker314)
  (clean shot402)
  (clean shot48)
  (clean shot238)
  (empty shaker314)
  (empty shot402)
  (empty shot48)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient144)
  (cocktail-part2 cocktail110 ingredient252)
)
 (:goal
  (and
      (contains shot402 cocktail110)
      (contains shot48 cocktail110)
)))
