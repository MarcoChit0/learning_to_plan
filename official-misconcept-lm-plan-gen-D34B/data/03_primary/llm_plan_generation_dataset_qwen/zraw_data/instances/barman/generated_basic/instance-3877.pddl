(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot51 shot376 - shot
      ingredient192 ingredient234 ingredient150 - ingredient
      cocktail404 - cocktail
      dispenser186 dispenser374 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot51)
  (ontable shot376)
  (dispenses dispenser186 ingredient192)
  (dispenses dispenser374 ingredient234)
  (dispenses dispenser331 ingredient150)
  (clean shaker318)
  (clean shot51)
  (clean shot376)
  (empty shaker318)
  (empty shot51)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail404 ingredient192)
  (cocktail-part2 cocktail404 ingredient150)
)
 (:goal
  (and
      (contains shot51 cocktail404)
)))
