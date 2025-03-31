(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot458 - shot
      ingredient137 ingredient143 - ingredient
      cocktail188 - cocktail
      dispenser191 dispenser169 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot458)
  (dispenses dispenser191 ingredient137)
  (dispenses dispenser169 ingredient143)
  (clean shaker331)
  (clean shot458)
  (empty shaker331)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail188 ingredient143)
  (cocktail-part2 cocktail188 ingredient137)
)
 (:goal
  (and
      (contains shot458 cocktail188)
)))
