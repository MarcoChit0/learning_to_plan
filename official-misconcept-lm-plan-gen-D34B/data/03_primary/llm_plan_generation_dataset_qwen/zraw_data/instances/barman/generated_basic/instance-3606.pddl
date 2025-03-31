(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot126 shot242 shot289 shot121 - shot
      ingredient444 ingredient175 ingredient131 - ingredient
      cocktail446 - cocktail
      dispenser124 dispenser270 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot126)
  (ontable shot242)
  (ontable shot289)
  (ontable shot121)
  (dispenses dispenser124 ingredient444)
  (dispenses dispenser270 ingredient175)
  (dispenses dispenser156 ingredient131)
  (clean shaker330)
  (clean shot126)
  (clean shot242)
  (clean shot289)
  (clean shot121)
  (empty shaker330)
  (empty shot126)
  (empty shot242)
  (empty shot289)
  (empty shot121)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient131)
  (cocktail-part2 cocktail446 ingredient175)
)
 (:goal
  (and
      (contains shot126 cocktail446)
      (contains shot242 cocktail446)
      (contains shot289 ingredient175)
)))
