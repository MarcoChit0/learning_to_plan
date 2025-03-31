(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot494 shot202 shot373 - shot
      ingredient423 ingredient105 ingredient275 ingredient336 - ingredient
      cocktail1 - cocktail
      dispenser286 dispenser302 dispenser82 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot494)
  (ontable shot202)
  (ontable shot373)
  (dispenses dispenser286 ingredient423)
  (dispenses dispenser302 ingredient105)
  (dispenses dispenser82 ingredient275)
  (dispenses dispenser458 ingredient336)
  (clean shaker132)
  (clean shot494)
  (clean shot202)
  (clean shot373)
  (empty shaker132)
  (empty shot494)
  (empty shot202)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient105)
  (cocktail-part2 cocktail1 ingredient275)
)
 (:goal
  (and
      (contains shot494 cocktail1)
      (contains shot202 cocktail1)
)))
