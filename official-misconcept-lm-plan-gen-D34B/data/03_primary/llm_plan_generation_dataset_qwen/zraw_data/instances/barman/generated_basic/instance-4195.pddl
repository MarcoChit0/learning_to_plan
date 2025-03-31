(define (problem prob)
 (:domain barman)
 (:objects 
      shaker168 - shaker
      left right - hand
      shot220 - shot
      ingredient52 ingredient154 ingredient152 ingredient255 - ingredient
      cocktail39 - cocktail
      dispenser80 dispenser37 dispenser336 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker168)
  (ontable shot220)
  (dispenses dispenser80 ingredient52)
  (dispenses dispenser37 ingredient154)
  (dispenses dispenser336 ingredient152)
  (dispenses dispenser71 ingredient255)
  (clean shaker168)
  (clean shot220)
  (empty shaker168)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker168 l0)
  (shaker-level shaker168 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient52)
  (cocktail-part2 cocktail39 ingredient152)
)
 (:goal
  (and
      (contains shot220 cocktail39)
)))
