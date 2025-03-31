(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot410 shot25 - shot
      ingredient237 ingredient470 ingredient359 - ingredient
      cocktail228 - cocktail
      dispenser173 dispenser252 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot410)
  (ontable shot25)
  (dispenses dispenser173 ingredient237)
  (dispenses dispenser252 ingredient470)
  (dispenses dispenser95 ingredient359)
  (clean shaker187)
  (clean shot410)
  (clean shot25)
  (empty shaker187)
  (empty shot410)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient470)
  (cocktail-part2 cocktail228 ingredient237)
)
 (:goal
  (and
      (contains shot410 cocktail228)
)))
