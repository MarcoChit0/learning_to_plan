(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot212 shot246 shot195 - shot
      ingredient7 ingredient78 - ingredient
      cocktail183 - cocktail
      dispenser174 dispenser300 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot212)
  (ontable shot246)
  (ontable shot195)
  (dispenses dispenser174 ingredient7)
  (dispenses dispenser300 ingredient78)
  (clean shaker426)
  (clean shot212)
  (clean shot246)
  (clean shot195)
  (empty shaker426)
  (empty shot212)
  (empty shot246)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient78)
  (cocktail-part2 cocktail183 ingredient7)
)
 (:goal
  (and
      (contains shot212 cocktail183)
      (contains shot246 cocktail183)
)))
