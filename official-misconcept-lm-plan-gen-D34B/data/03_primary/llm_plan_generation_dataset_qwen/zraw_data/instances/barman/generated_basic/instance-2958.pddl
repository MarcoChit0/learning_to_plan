(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot188 - shot
      ingredient368 ingredient144 ingredient104 ingredient201 - ingredient
      cocktail111 - cocktail
      dispenser148 dispenser249 dispenser357 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot188)
  (dispenses dispenser148 ingredient368)
  (dispenses dispenser249 ingredient144)
  (dispenses dispenser357 ingredient104)
  (dispenses dispenser13 ingredient201)
  (clean shaker139)
  (clean shot188)
  (empty shaker139)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient368)
  (cocktail-part2 cocktail111 ingredient201)
)
 (:goal
  (and
      (contains shot188 cocktail111)
)))
