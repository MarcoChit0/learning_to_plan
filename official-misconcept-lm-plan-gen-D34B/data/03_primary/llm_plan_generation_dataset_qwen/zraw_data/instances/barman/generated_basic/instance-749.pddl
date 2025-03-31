(define (problem prob)
 (:domain barman)
 (:objects 
      shaker93 - shaker
      left right - hand
      shot316 shot285 - shot
      ingredient161 ingredient39 ingredient21 - ingredient
      cocktail1 - cocktail
      dispenser352 dispenser353 dispenser488 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker93)
  (ontable shot316)
  (ontable shot285)
  (dispenses dispenser352 ingredient161)
  (dispenses dispenser353 ingredient39)
  (dispenses dispenser488 ingredient21)
  (clean shaker93)
  (clean shot316)
  (clean shot285)
  (empty shaker93)
  (empty shot316)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker93 l0)
  (shaker-level shaker93 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient21)
  (cocktail-part2 cocktail1 ingredient161)
)
 (:goal
  (and
      (contains shot316 cocktail1)
)))
