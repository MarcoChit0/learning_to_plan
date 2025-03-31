(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot485 shot287 - shot
      ingredient421 ingredient218 ingredient373 ingredient149 - ingredient
      cocktail11 - cocktail
      dispenser62 dispenser442 dispenser11 dispenser435 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot485)
  (ontable shot287)
  (dispenses dispenser62 ingredient421)
  (dispenses dispenser442 ingredient218)
  (dispenses dispenser11 ingredient373)
  (dispenses dispenser435 ingredient149)
  (clean shaker10)
  (clean shot485)
  (clean shot287)
  (empty shaker10)
  (empty shot485)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail11 ingredient218)
  (cocktail-part2 cocktail11 ingredient149)
)
 (:goal
  (and
      (contains shot485 cocktail11)
)))
