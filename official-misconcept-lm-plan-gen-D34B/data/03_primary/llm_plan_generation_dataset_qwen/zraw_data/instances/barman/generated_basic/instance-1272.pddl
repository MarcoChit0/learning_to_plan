(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot108 - shot
      ingredient251 ingredient84 ingredient105 ingredient438 - ingredient
      cocktail12 - cocktail
      dispenser259 dispenser272 dispenser232 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot108)
  (dispenses dispenser259 ingredient251)
  (dispenses dispenser272 ingredient84)
  (dispenses dispenser232 ingredient105)
  (dispenses dispenser245 ingredient438)
  (clean shaker314)
  (clean shot108)
  (empty shaker314)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient438)
  (cocktail-part2 cocktail12 ingredient105)
)
 (:goal
  (and
      (contains shot108 cocktail12)
)))
