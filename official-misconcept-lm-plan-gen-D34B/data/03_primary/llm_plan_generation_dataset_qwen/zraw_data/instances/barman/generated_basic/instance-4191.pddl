(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot438 - shot
      ingredient481 ingredient229 ingredient490 ingredient449 - ingredient
      cocktail334 - cocktail
      dispenser0 dispenser348 dispenser317 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot438)
  (dispenses dispenser0 ingredient481)
  (dispenses dispenser348 ingredient229)
  (dispenses dispenser317 ingredient490)
  (dispenses dispenser471 ingredient449)
  (clean shaker251)
  (clean shot438)
  (empty shaker251)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient481)
  (cocktail-part2 cocktail334 ingredient229)
)
 (:goal
  (and
      (contains shot438 cocktail334)
)))
