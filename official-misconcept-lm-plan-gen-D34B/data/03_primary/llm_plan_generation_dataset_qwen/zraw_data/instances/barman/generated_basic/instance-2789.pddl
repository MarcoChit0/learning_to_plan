(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot104 - shot
      ingredient421 ingredient88 ingredient222 ingredient420 - ingredient
      cocktail431 - cocktail
      dispenser158 dispenser308 dispenser253 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot104)
  (dispenses dispenser158 ingredient421)
  (dispenses dispenser308 ingredient88)
  (dispenses dispenser253 ingredient222)
  (dispenses dispenser222 ingredient420)
  (clean shaker401)
  (clean shot104)
  (empty shaker401)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient222)
  (cocktail-part2 cocktail431 ingredient421)
)
 (:goal
  (and
      (contains shot104 cocktail431)
)))
