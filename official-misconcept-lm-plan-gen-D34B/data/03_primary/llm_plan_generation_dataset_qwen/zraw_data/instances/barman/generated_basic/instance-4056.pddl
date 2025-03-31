(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot300 shot166 - shot
      ingredient381 ingredient102 - ingredient
      cocktail444 - cocktail
      dispenser316 dispenser438 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot300)
  (ontable shot166)
  (dispenses dispenser316 ingredient381)
  (dispenses dispenser438 ingredient102)
  (clean shaker126)
  (clean shot300)
  (clean shot166)
  (empty shaker126)
  (empty shot300)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail444 ingredient381)
  (cocktail-part2 cocktail444 ingredient102)
)
 (:goal
  (and
      (contains shot300 cocktail444)
)))
