(define (problem prob)
 (:domain barman)
 (:objects 
      shaker182 - shaker
      left right - hand
      shot150 shot393 - shot
      ingredient242 ingredient122 ingredient326 ingredient490 - ingredient
      cocktail256 - cocktail
      dispenser138 dispenser16 dispenser11 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker182)
  (ontable shot150)
  (ontable shot393)
  (dispenses dispenser138 ingredient242)
  (dispenses dispenser16 ingredient122)
  (dispenses dispenser11 ingredient326)
  (dispenses dispenser135 ingredient490)
  (clean shaker182)
  (clean shot150)
  (clean shot393)
  (empty shaker182)
  (empty shot150)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker182 l0)
  (shaker-level shaker182 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient490)
  (cocktail-part2 cocktail256 ingredient242)
)
 (:goal
  (and
      (contains shot150 cocktail256)
)))
